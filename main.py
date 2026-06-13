# ============================================================
# FINANCIAL ADVISOR AGENT — Stage 4
# Portfolio tracking + News + Daily Telegram Briefing
# Deployed on Railway (always-on)
# ============================================================
#
# WHAT'S NEW vs STAGE 3:
#   - Portfolio stored in portfolio.json (list of {symbol, shares, cost_basis})
#   - add_holding / remove_holding tools — agent edits portfolio via chat
#   - get_market_news tool — fetches headlines for a symbol or general market
#   - daily_briefing() — builds a full summary and sends to Telegram
#   - APScheduler runs daily_briefing() once a day automatically
#   - /chat endpoint still works exactly like Stage 3 for normal conversation
#
# FILES IN THIS PROJECT:
#   main.py            ← this file
#   requirements.txt   ← dependencies for Railway
#   portfolio.json     ← created automatically, stores your holdings
#   memory.json        ← created automatically, stores your profile
#
# ENVIRONMENT VARIABLES (set these in Railway dashboard):
#   ANTHROPIC_API_KEY
#   TELEGRAM_BOT_TOKEN
#   TELEGRAM_CHAT_ID
#   BRIEFING_HOUR_UTC_1 (e.g. "6" for 6 AM UTC ≈ 9 AM Riyadh — morning briefing)
#   BRIEFING_HOUR_UTC_2 (e.g. "16" for 4 PM UTC ≈ 7 PM Riyadh — evening briefing)
# ============================================================

import json
import os
import re
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import anthropic
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

# ── CONFIG (from environment variables) ──────────────────────
API_KEY            = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN     = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
BRIEFING_HOUR_UTC_1 = int(os.environ.get("BRIEFING_HOUR_UTC_1", "6"))   # ~9 AM Riyadh
BRIEFING_HOUR_UTC_2 = int(os.environ.get("BRIEFING_HOUR_UTC_2", "16"))  # ~7 PM Riyadh

MODEL = "claude-sonnet-4-5"

DATA_DIR = Path("/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE    = DATA_DIR / "memory.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TELEGRAM_HISTORY_FILE = DATA_DIR / "telegram_history.json"


# ── STORAGE HELPERS ───────────────────────────────────────────
def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def load_memory() -> dict:
    return load_json(MEMORY_FILE, {})

def save_memory(profile: dict):
    save_json(MEMORY_FILE, profile)

def load_portfolio() -> list:
    """Portfolio is a list of: {"symbol": "AAPL", "shares": 10, "cost_basis": 150.0}"""
    return load_json(PORTFOLIO_FILE, [])

def save_portfolio(portfolio: list):
    save_json(PORTFOLIO_FILE, portfolio)


# ── PRICE / NEWS / FX HELPERS (same as Stage 3, plus news) ────
def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

def get_price(symbol: str) -> dict:
    try:
        data = fetch_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d")
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        prev_close = meta.get("previousClose", price)
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        return {
            "symbol": symbol.upper(),
            "name": meta.get("shortName") or symbol,
            "price": round(price, 2),
            "previous_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "currency": meta.get("currency", "USD")
        }
    except Exception as e:
        return {"error": f"Could not fetch price for '{symbol}'. ({e})"}

def calculate_compound(principal: float, annual_rate_pct: float,
                        years: float, monthly_contribution: float = 0) -> dict:
    r = (annual_rate_pct / 100) / 12
    n = int(years * 12)
    fv_principal = principal * ((1 + r) ** n)
    if r > 0 and monthly_contribution > 0:
        fv_contributions = monthly_contribution * (((1 + r) ** n - 1) / r)
    else:
        fv_contributions = monthly_contribution * n
    total = fv_principal + fv_contributions
    total_invested = principal + (monthly_contribution * n)
    total_gain = total - total_invested
    return {
        "principal": round(principal, 2),
        "monthly_contribution": round(monthly_contribution, 2),
        "annual_rate_pct": annual_rate_pct,
        "years": years,
        "future_value": round(total, 2),
        "total_invested": round(total_invested, 2),
        "total_gain": round(total_gain, 2),
        "gain_pct": round((total_gain / total_invested * 100) if total_invested else 0, 1)
    }

def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    try:
        data = fetch_json(f"https://open.er-api.com/v6/latest/{from_currency.upper()}")
        to = to_currency.upper()
        if to not in data["rates"]:
            return {"error": f"Currency '{to}' not found."}
        rate = data["rates"][to]
        return {"from": from_currency.upper(), "to": to, "rate": round(rate, 4),
                "example": f"1 {from_currency.upper()} = {round(rate, 4)} {to}"}
    except Exception as e:
        return {"error": f"Could not fetch exchange rate. ({e})"}

def get_market_news(query: str = "stock market") -> dict:
    """
    Fetch recent news headlines using Yahoo Finance's search endpoint (free, no key).
    query can be a symbol (e.g. 'AAPL') or general term (e.g. 'market').
    """
    try:
        q = urllib.parse.quote(query)
        data = fetch_json(f"https://query1.finance.yahoo.com/v1/finance/search?q={q}&newsCount=6")
        items = data.get("news", [])[:6]
        headlines = [{"title": i.get("title"), "publisher": i.get("publisher"),
                       "link": i.get("link")} for i in items]
        return {"query": query, "headlines": headlines}
    except Exception as e:
        return {"error": f"Could not fetch news. ({e})"}


# ── PORTFOLIO TOOLS ────────────────────────────────────────────
def add_holding(symbol: str, shares: float, cost_basis: float) -> dict:
    portfolio = load_portfolio()
    symbol = symbol.upper()
    for h in portfolio:
        if h["symbol"] == symbol:
            # Merge: weighted average cost basis
            total_shares = h["shares"] + shares
            total_cost = h["shares"] * h["cost_basis"] + shares * cost_basis
            h["shares"] = total_shares
            h["cost_basis"] = round(total_cost / total_shares, 2)
            save_portfolio(portfolio)
            return {"status": "updated", "holding": h}
    portfolio.append({"symbol": symbol, "shares": shares, "cost_basis": cost_basis})
    save_portfolio(portfolio)
    return {"status": "added", "holding": portfolio[-1]}

def remove_holding(symbol: str) -> dict:
    portfolio = load_portfolio()
    symbol = symbol.upper()
    new_portfolio = [h for h in portfolio if h["symbol"] != symbol]
    if len(new_portfolio) == len(portfolio):
        return {"error": f"{symbol} not found in portfolio."}
    save_portfolio(new_portfolio)
    return {"status": "removed", "symbol": symbol}

def get_portfolio_summary() -> dict:
    portfolio = load_portfolio()
    if not portfolio:
        return {"holdings": [], "message": "Portfolio is empty."}
    results = []
    total_value = 0
    total_cost = 0
    for h in portfolio:
        price_data = get_price(h["symbol"])
        if "error" in price_data:
            results.append({**h, "error": price_data["error"]})
            continue
        current_price = price_data["price"]
        value = current_price * h["shares"]
        cost = h["cost_basis"] * h["shares"]
        gain = value - cost
        gain_pct = (gain / cost * 100) if cost else 0
        total_value += value
        total_cost += cost
        results.append({
            "symbol": h["symbol"],
            "shares": h["shares"],
            "cost_basis": h["cost_basis"],
            "current_price": current_price,
            "value": round(value, 2),
            "gain": round(gain, 2),
            "gain_pct": round(gain_pct, 2),
            "day_change_pct": price_data.get("change_pct", 0)
        })
    total_gain = total_value - total_cost
    return {
        "holdings": results,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_gain": round(total_gain, 2),
        "total_gain_pct": round((total_gain / total_cost * 100) if total_cost else 0, 2)
    }


# ── TOOL DEFINITIONS FOR CLAUDE ────────────────────────────────
TOOLS = [
    {
        "name": "get_price",
        "description": "Get the current live price of a stock, crypto, or commodity. Use for any price question.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL, BTC-USD, GC=F"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "calculate_compound",
        "description": "Calculate compound interest / investment growth over time with optional monthly contributions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "annual_rate_pct": {"type": "number"},
                "years": {"type": "number"},
                "monthly_contribution": {"type": "number"}
            },
            "required": ["principal", "annual_rate_pct", "years"]
        }
    },
    {
        "name": "get_exchange_rate",
        "description": "Get the current exchange rate between two currencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"}
            },
            "required": ["from_currency", "to_currency"]
        }
    },
    {
        "name": "get_market_news",
        "description": "Get recent news headlines for a stock symbol or general market topic. Use when discussing earnings, market news, or 'what's happening with X'.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Symbol (e.g. AAPL) or topic (e.g. 'stock market', 'Fed interest rates')"}},
            "required": ["query"]
        }
    },
    {
        "name": "add_holding",
        "description": "Add or update a stock holding in the user's portfolio. Use when the user says they bought/own shares of something.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
                "shares": {"type": "number", "description": "Number of shares owned"},
                "cost_basis": {"type": "number", "description": "Average price per share paid"}
            },
            "required": ["symbol", "shares", "cost_basis"]
        }
    },
    {
        "name": "remove_holding",
        "description": "Remove a stock from the user's portfolio entirely. Use when the user says they sold all shares of something.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "name": "get_portfolio_summary",
        "description": "Get the user's full portfolio with current values, gains/losses for each holding and in total. Use whenever the user asks about their portfolio, gains, or losses.",
        "input_schema": {"type": "object", "properties": {}}
    }
]

TOOL_FUNCTIONS = {
    "get_price": get_price,
    "calculate_compound": calculate_compound,
    "get_exchange_rate": get_exchange_rate,
    "get_market_news": get_market_news,
    "add_holding": add_holding,
    "remove_holding": remove_holding,
    "get_portfolio_summary": get_portfolio_summary,
}

def run_tool(name: str, inputs: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(func(**inputs))


# ── SYSTEM PROMPT ───────────────────────────────────────────────
def memory_to_text(profile: dict) -> str:
    if not profile:
        return "No profile yet — this is a new user."
    lines = ["What I already know about this user:"]
    for key, value in profile.items():
        if isinstance(value, list):
            lines.append(f"  - {key}: {', '.join(value)}")
        else:
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines)

def build_system_prompt(profile: dict) -> str:
    memory_text = memory_to_text(profile)
    return f"""You are a knowledgeable, friendly, and practical personal financial advisor.

{memory_text}

YOUR JOB:
- Give tailored financial advice based on what you know about this user
- If this is a new user (no profile), warmly introduce yourself and ask 2-3 questions about their situation
- Use tools proactively for prices, news, calculations, exchange rates, and portfolio data — never guess
- When the user mentions buying/owning/selling stocks, use add_holding / remove_holding to keep their portfolio updated
- Be encouraging but honest about risks
- Keep responses concise and focused
- Always remind the user you are an AI, not a licensed financial advisor, for major decisions

MEMORY EXTRACTION:
After each user message, if you learn new facts about the user's finances (NOT portfolio holdings — those go through add_holding/remove_holding), include this block at the END of your response:

<memory_update>
{{
  "name": "Ahmed",
  "monthly_income": "15000 SAR",
  "goals": ["buy a house"],
  "debts": ["car loan 800 SAR/month"],
  "risk_tolerance": "moderate",
  "notes": ["has 3 dependents"]
}}
</memory_update>

Only include fields you actually learned. Only add this block if there is new information. The user will NOT see this block."""


def extract_and_update_memory(reply_text: str, profile: dict):
    pattern = r"<memory_update>(.*?)</memory_update>"
    match = re.search(pattern, reply_text, re.DOTALL)
    if match:
        try:
            new_facts = json.loads(match.group(1).strip())
            for key, value in new_facts.items():
                if isinstance(value, list) and isinstance(profile.get(key), list):
                    existing = profile[key]
                    for item in value:
                        if item not in existing:
                            existing.append(item)
                    profile[key] = existing
                else:
                    profile[key] = value
        except json.JSONDecodeError:
            pass
    clean_reply = re.sub(pattern, "", reply_text, flags=re.DOTALL).strip()
    return clean_reply, profile


# ── AGENT LOOP (shared by /chat and daily briefing) ──────────────
def run_agent(messages: list, system_prompt: str) -> str:
    """Runs the full tool-use loop and returns final text."""
    client = anthropic.Anthropic(api_key=API_KEY)
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        if response.stop_reason == "tool_use":
            messages = messages + [{"role": "assistant", "content": response.content}]
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages = messages + [{"role": "user", "content": tool_results}]
            continue

        return "Sorry, something went wrong. Please try again."


# ── TELEGRAM ─────────────────────────────────────────────────────
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Telegram has a 4096 char limit per message — split if needed
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)] or [text]
    for chunk in chunks:
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown"
        }).encode()
        req = urllib.request.Request(url, data=data)
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"Telegram send error: {e}")


# ── USER BRIEFING PREFERENCES ────────────────────────────────────
# Edit these anytime to change what the briefing focuses on.
SECTORS_OF_INTEREST = ["technology", "oil & energy", "retail", "transportation", "consumer staples"]
EXCLUDED_AREAS = [
    "media/entertainment (e.g. Netflix-type companies)",
    "defense-reliant companies (e.g. Boeing, Palantir)",
    "insurance",
    "alcohol & beverages",
    "bonds",
    "cryptocurrency"
]

def daily_briefing(slot: str = "morning"):
    """
    Generates and sends a market briefing via Telegram.
    slot: "morning" (pre-market/open focus, 9 AM Riyadh) or
          "evening" (recap of US session + TASI close, 7 PM Riyadh)
    """
    print(f"[{datetime.now()}] Running {slot} briefing...")
    profile = load_memory()
    system_prompt = build_system_prompt(profile)

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    sectors_str = ", ".join(SECTORS_OF_INTEREST)
    excluded_str = "; ".join(EXCLUDED_AREAS)

    if slot == "morning":
        time_context = (
            "This is the MORNING briefing (9 AM Riyadh time). Focus on: overnight US market close summary "
            "(since US markets just closed a few hours ago), today's TASI (Saudi market) outlook/open, "
            "and what to watch for during today's trading sessions (TASI open + US pre-market)."
        )
    else:
        time_context = (
            "This is the EVENING briefing (7 PM Riyadh time). Focus on: TASI (Saudi market) closing summary "
            "for today, US market pre-market/early session setup (US markets are about to open or just opened), "
            "and key events/earnings expected during the US session tonight."
        )

    prompt = f"""Today's date is {today_str}. Generate my detailed financial briefing.

{time_context}

USER CONTEXT:
- Portfolio is mostly US-listed stocks, with some interest in TASI (Saudi stock market).
- Sectors of interest: {sectors_str}.
- NOT interested in: {excluded_str}. Do not bring up or recommend anything in these areas unless directly relevant to a holding the user already owns.

STRUCTURE — cover all of these in detail:

1. **Portfolio summary** — use get_portfolio_summary. Give total gain/loss, and a breakdown per holding (value, day change %, overall gain %). Comment briefly on what's driving notable moves.

2. **US Market Overview** — use get_market_news with query "stock market" and also "US stock market today". Summarize the major indices direction (S&P 500, Nasdaq, Dow) and the main narrative driving the day (Fed, earnings season, macro data, etc).

3. **TASI / Saudi Market Overview** — use get_market_news with query "TASI Saudi stock market" for the latest on the Saudi market — index level, notable movers, any major Saudi economic news (oil policy, Aramco, PIF, etc).

4. **Sector Watch** — for each sector of interest ({sectors_str}), use get_market_news to check for major headlines today (e.g. "oil prices news today", "tech stocks news today", "retail sector news"). Highlight anything significant — especially earnings reports, M&A, regulatory news, or major price-moving events.

5. **News on Your Holdings** — for each stock in the portfolio, use get_market_news with that symbol. Flag earnings dates, analyst rating changes, or major news. Be specific.

6. **Things to Watch / Short-Term Considerations** — based on everything above, give 2-4 specific, detailed observations on potential short-term opportunities or risks (within sectors of interest, and respecting the exclusions above). Frame these as "worth watching" / "consider researching further" — not direct buy/sell instructions.

FORMAT:
- Use simple Telegram markdown (*bold*, bullet points with -, no # headers)
- This is the DETAILED version — thoroughness matters more than brevity, but stay organized with clear section breaks
- Start with a one-line header: the date + which briefing (Morning/Evening)
- End with a one-line reminder that this is AI-generated analysis, not financial advice"""

    messages = [{"role": "user", "content": prompt}]
    reply = run_agent(messages, system_prompt)

    clean_reply, updated_profile = extract_and_update_memory(reply, profile)
    save_memory(updated_profile)

    send_telegram_message(clean_reply)
    print(f"[{datetime.now()}] {slot.capitalize()} briefing sent.")


def morning_briefing():
    daily_briefing("morning")

def evening_briefing():
    daily_briefing("evening")


# ── FASTAPI APP ────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/chat")
async def chat(request: ChatRequest):
    profile = load_memory()
    system_prompt = build_system_prompt(profile)
    reply = run_agent(request.messages, system_prompt)
    clean_reply, updated_profile = extract_and_update_memory(reply, profile)
    save_memory(updated_profile)
    return {"reply": clean_reply, "profile": updated_profile}


# ── TELEGRAM TWO-WAY CHAT ─────────────────────────────────────────
def load_telegram_history() -> list:
    return load_json(TELEGRAM_HISTORY_FILE, [])

def save_telegram_history(history: list):
    # Keep only the last 30 messages to avoid unbounded growth
    save_json(TELEGRAM_HISTORY_FILE, history[-30:])

@app.post("/telegram/webhook")
async def telegram_webhook(update: dict):
    """
    Receives incoming Telegram messages. Configure this URL as your bot's webhook:
    https://api.telegram.org/bot<TOKEN>/setWebhook?url=<YOUR_RAILWAY_URL>/telegram/webhook
    """
    message = update.get("message", {})
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = message.get("text", "")

    # Only respond to the configured chat (security: ignore other chats)
    if chat_id != TELEGRAM_CHAT_ID or not text:
        return {"ok": True}

    # Special commands
    if text.strip() == "/reset":
        save_telegram_history([])
        send_telegram_message("Conversation history cleared. Starting fresh!")
        return {"ok": True}

    profile = load_memory()
    system_prompt = build_system_prompt(profile)

    history = load_telegram_history()
    history.append({"role": "user", "content": text})

    reply = run_agent(history, system_prompt)
    clean_reply, updated_profile = extract_and_update_memory(reply, profile)
    save_memory(updated_profile)

    history.append({"role": "assistant", "content": clean_reply})
    save_telegram_history(history)

    send_telegram_message(clean_reply)
    return {"ok": True}

@app.get("/memory")
async def get_memory():
    return load_memory()

@app.delete("/memory")
async def clear_memory():
    save_memory({})
    return {"status": "memory cleared"}

@app.get("/portfolio")
async def get_portfolio():
    return get_portfolio_summary()

@app.post("/briefing/test")
async def test_briefing(slot: str = "morning"):
    """Manually trigger a briefing — useful for testing. slot=morning or evening"""
    daily_briefing(slot)
    return {"status": "briefing sent", "slot": slot}

@app.get("/")
async def root():
    return {"status": "Financial Advisor Agent is running", "time": str(datetime.now())}


# ── SCHEDULER ──────────────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(morning_briefing, "cron", hour=BRIEFING_HOUR_UTC_1, minute=0)
scheduler.add_job(evening_briefing, "cron", hour=BRIEFING_HOUR_UTC_2, minute=0)
scheduler.start()
print(f"Scheduler started — morning briefing at {BRIEFING_HOUR_UTC_1}:00 UTC, evening at {BRIEFING_HOUR_UTC_2}:00 UTC")


# ── RUN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
