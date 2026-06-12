# ============================================================
# FINANCIAL ADVISOR AGENT — server.py (Stage 3: Tools)
# ============================================================
# NEW IN STAGE 3:
#   - Tool use: agent can call functions mid-conversation
#   - get_price: live stock/crypto/commodity prices
#   - calculate_compound: savings & investment projections
#   - get_exchange_rate: live currency conversion
#
# HOW TOOL USE WORKS:
#   1. You send a message
#   2. Claude decides if it needs a tool
#   3. If yes, it returns a "tool_use" block instead of text
#   4. Our code runs the tool and sends the result back
#   5. Claude reads the result and forms the final answer
#   This loop runs inside /chat — browser sees nothing different
# ============================================================

import json
import math
import re
import urllib.request
from pathlib import Path

import anthropic
import nest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

nest_asyncio.apply()

# ── CONFIG ───────────────────────────────────────────────────
API_KEY      = "YOUR_API_KEY_HERE"   # ← your Anthropic key
MODEL        = "claude-sonnet-4-5"

# Google Drive path (Stage 2 fix — keeps memory across Colab restarts)
MEMORY_FILE  = Path("/content/drive/MyDrive/financial_agent/memory.json")
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── TOOL DEFINITIONS ─────────────────────────────────────────
# This is the list we send to Claude so it knows what tools exist.
# Claude reads the "description" fields to decide when to use each tool.

TOOLS = [
    {
        "name": "get_price",
        "description": (
            "Get the current live price of a stock, cryptocurrency, or commodity. "
            "Use this whenever the user asks about the price of any asset: "
            "stocks (e.g. AAPL, TSLA, ARAMCO), crypto (BTC, ETH), or commodities (gold, oil). "
            "Always use this tool for price questions — never guess a price from memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The ticker symbol. Examples: AAPL, BTC-USD, GC=F (gold), CL=F (oil)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "calculate_compound",
        "description": (
            "Calculate compound interest / investment growth over time. "
            "Use this when the user wants to know: how much their savings will grow, "
            "how long to reach a goal, or the impact of monthly contributions. "
            "Always use real numbers the user has shared."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "principal":          {"type": "number", "description": "Starting amount (SAR or any currency)"},
                "annual_rate_pct":    {"type": "number", "description": "Annual interest/return rate as a percentage, e.g. 7 for 7%"},
                "years":              {"type": "number", "description": "Number of years"},
                "monthly_contribution":{"type": "number", "description": "Optional monthly deposit amount (default 0)"}
            },
            "required": ["principal", "annual_rate_pct", "years"]
        }
    },
    {
        "name": "get_exchange_rate",
        "description": (
            "Get the current exchange rate between two currencies. "
            "Use this when the user asks about currency conversion, e.g. USD to SAR, EUR to SAR, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "Source currency code, e.g. USD"},
                "to_currency":   {"type": "string", "description": "Target currency code, e.g. SAR"}
            },
            "required": ["from_currency", "to_currency"]
        }
    }
]

# ── TOOL IMPLEMENTATIONS ─────────────────────────────────────
# These are the actual Python functions that run when Claude calls a tool.

def get_price(symbol: str) -> dict:
    """Fetch live price from Yahoo Finance (no API key needed)."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        meta  = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice") or meta.get("previousClose")
        currency = meta.get("currency", "USD")
        name     = meta.get("shortName") or symbol
        return {
            "symbol":   symbol.upper(),
            "name":     name,
            "price":    round(price, 2),
            "currency": currency
        }
    except Exception as e:
        return {"error": f"Could not fetch price for '{symbol}'. Try a different symbol. ({e})"}


def calculate_compound(principal: float, annual_rate_pct: float,
                        years: float, monthly_contribution: float = 0) -> dict:
    """
    Compound interest with optional monthly contributions.
    Formula: FV = P*(1+r)^n + PMT * [((1+r)^n - 1) / r]
    where r = monthly rate, n = months
    """
    r = (annual_rate_pct / 100) / 12   # monthly rate
    n = int(years * 12)                 # total months

    # Future value of lump sum
    fv_principal = principal * ((1 + r) ** n)

    # Future value of monthly contributions
    if r > 0 and monthly_contribution > 0:
        fv_contributions = monthly_contribution * (((1 + r) ** n - 1) / r)
    else:
        fv_contributions = monthly_contribution * n

    total = fv_principal + fv_contributions
    total_invested = principal + (monthly_contribution * n)
    total_gain = total - total_invested

    return {
        "principal":           round(principal, 2),
        "monthly_contribution": round(monthly_contribution, 2),
        "annual_rate_pct":     annual_rate_pct,
        "years":               years,
        "future_value":        round(total, 2),
        "total_invested":      round(total_invested, 2),
        "total_gain":          round(total_gain, 2),
        "gain_pct":            round((total_gain / total_invested * 100) if total_invested else 0, 1)
    }


def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """Fetch live exchange rate from a free public API."""
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        to = to_currency.upper()
        if to not in data["rates"]:
            return {"error": f"Currency '{to}' not found."}
        rate = data["rates"][to]
        return {
            "from":      from_currency.upper(),
            "to":        to,
            "rate":      round(rate, 4),
            "example":   f"1 {from_currency.upper()} = {round(rate, 4)} {to}"
        }
    except Exception as e:
        return {"error": f"Could not fetch exchange rate. ({e})"}


# Map tool names → functions so we can call them dynamically
TOOL_FUNCTIONS = {
    "get_price":          get_price,
    "calculate_compound": calculate_compound,
    "get_exchange_rate":  get_exchange_rate,
}

def run_tool(name: str, inputs: dict) -> str:
    """Run a tool by name and return its result as a JSON string."""
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = func(**inputs)
    return json.dumps(result)


# ── MEMORY HELPERS ───────────────────────────────────────────
def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {}

def save_memory(profile: dict):
    MEMORY_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False))

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
- If this is a new user (no profile), warmly introduce yourself and ask 2-3 questions to understand their situation
- Give concrete, actionable advice — not vague generalities
- Use your tools proactively: whenever prices, calculations, or exchange rates are relevant, use the tools — don't guess
- Be encouraging but honest about risks
- Keep responses concise and focused
- Always remind the user you are an AI, not a licensed financial advisor, for major decisions

TOOLS YOU HAVE:
- get_price: live price of any stock, crypto, or commodity
- calculate_compound: investment/savings growth projections
- get_exchange_rate: live currency conversion rates

MEMORY EXTRACTION:
After each user message, if you learn new facts about the user's finances, include this block at the END of your response:

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

def extract_and_update_memory(reply_text: str, profile: dict) -> tuple[str, dict]:
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


# ── FASTAPI APP ──────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list[dict]

# ── CHAT ENDPOINT WITH TOOL LOOP ─────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    profile = load_memory()
    system_prompt = build_system_prompt(profile)
    client = anthropic.Anthropic(api_key=API_KEY)

    messages = request.messages

    # ── AGENT LOOP ───────────────────────────────────────────
    # We loop because Claude might call multiple tools in one turn.
    # Each iteration: call Claude → if it wants a tool, run it and loop again
    # → if it returns text, we're done.

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system_prompt,
            tools=TOOLS,           # ← tell Claude what tools are available
            messages=messages
        )

        # Claude returned a final text answer — we're done
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            break

        # Claude wants to use one or more tools
        if response.stop_reason == "tool_use":
            # Add Claude's response (including tool_use blocks) to history
            messages = messages + [{"role": "assistant", "content": response.content}]

            # Run each tool Claude requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 Tool called: {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    print(f"  ✓  Result: {result}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result
                    })

            # Add tool results to history so Claude can read them
            messages = messages + [{"role": "user", "content": tool_results}]
            # Loop again — Claude will now form its final answer using the results
            continue

        # Unexpected stop reason — break safely
        final_text = "Sorry, something went wrong. Please try again."
        break

    # Extract memory updates and clean the reply
    clean_reply, updated_profile = extract_and_update_memory(final_text, profile)
    save_memory(updated_profile)

    return {"reply": clean_reply, "profile": updated_profile}


@app.get("/memory")
async def get_memory():
    return load_memory()

@app.delete("/memory")
async def clear_memory():
    save_memory({})
    return {"status": "memory cleared"}


# ── START ────────────────────────────────────────────────────
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print("=" * 50)
print(f"  Server running!")
print(f"  Public URL: {public_url.public_url}")
print(f"  Paste this into index.html → Connect")
print("=" * 50)

config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()
