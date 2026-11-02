import os
import json
import re
import html
import time
import hashlib
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Reusable high-speed HTTP Session with persistent connection pooling
http_session = requests.Session()
retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=40, max_retries=retries)
http_session.mount('https://', adapter)
http_session.mount('http://', adapter)
http_session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

# Server-level default key (Optional)
PRIMARY_API_KEY = (
    os.environ.get("NOVA_API_KEY") or 
    os.environ.get("GROQ_API_KEY") or 
    os.environ.get("GEMINI_API_KEY") or 
    os.environ.get("OPENAI_API_KEY") or 
    os.environ.get("CLAUDE_API_KEY") or 
    os.environ.get("AI_API_KEY") or 
    ""
).strip()

# ==============================================================================
# 1. PER-USER TOKEN TRACKING & 7-HOUR RESET ENGINE (MOBILE & DESKTOP IP AWARE)
# ==============================================================================
FREE_TIER_TOKEN_LIMIT = 20000        # 20,000 tokens per user cycle
WINDOW_DURATION_SECONDS = 7 * 3600  # 7-Hour sliding reset cycle (25,200 seconds)

class UserTokenTracker:
    def __init__(self):
        self.lock = threading.Lock()
        # IP -> { "tokens_used": int, "window_start": float, "first_seen": float, "request_count": int }
        self.usage_records = {}

    def get_client_ip(self, req):
        """Extract reliable IP address handling mobile cellular proxies & Cloudflare."""
        forwarded = req.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return req.headers.get("CF-Connecting-IP") or req.remote_addr or "127.0.0.1"

    def get_or_reset(self, ip):
        now = time.time()
        with self.lock:
            if ip not in self.usage_records:
                self.usage_records[ip] = {
                    "tokens_used": 0,
                    "window_start": now,
                    "first_seen": now,
                    "request_count": 0
                }
            else:
                data = self.usage_records[ip]
                elapsed = now - data["window_start"]
                if elapsed >= WINDOW_DURATION_SECONDS:
                    # 7-hour cycle elapsed -> automatically reset tokens to 0
                    data["tokens_used"] = 0
                    data["window_start"] = now
                    data["request_count"] = 0
            return dict(self.usage_records[ip])

    def consume_tokens(self, ip, token_count):
        now = time.time()
        with self.lock:
            if ip in self.usage_records:
                data = self.usage_records[ip]
                if now - data["window_start"] >= WINDOW_DURATION_SECONDS:
                    data["tokens_used"] = 0
                    data["window_start"] = now
                    data["request_count"] = 0
                data["tokens_used"] += max(0, int(token_count))
                data["request_count"] += 1

    def get_status(self, ip, has_personal_key=False):
        data = self.get_or_reset(ip)
        now = time.time()
        elapsed = now - data["window_start"]
        remaining_seconds = max(0, int(WINDOW_DURATION_SECONDS - elapsed))
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        resets_formatted = f"{hours}h {minutes}m"

        tokens_used = data["tokens_used"]
        tokens_remaining = max(0, FREE_TIER_TOKEN_LIMIT - tokens_used)
        pct_used = min(100.0, round((tokens_used / FREE_TIER_TOKEN_LIMIT) * 100, 1))

        # Securely mask IP for user display (e.g. 192.168.***.***)
        parts = ip.split(".")
        masked_ip = f"{parts[0]}.{parts[1]}.***.***" if len(parts) == 4 else (ip[:8] + "...")

        return {
            "ip": masked_ip,
            "has_personal_key": has_personal_key,
            "has_custom_key": has_personal_key,
            "tier": "Personal Key (BYOK Account)" if has_personal_key else "Default User (Shared Groq Engine · 20,000 tokens/7h)",
            "token_limit": "Personal Quota" if has_personal_key else FREE_TIER_TOKEN_LIMIT,
            "tokens_used": tokens_used,
            "tokens_remaining": "Personal Quota" if has_personal_key else tokens_remaining,
            "percent_used": 0 if has_personal_key else pct_used,
            "resets_in_seconds": remaining_seconds,
            "resets_formatted": resets_formatted,
            "cycle_hours": 7,
            "request_count": data["request_count"],
            "can_proceed": has_personal_key or (tokens_used < FREE_TIER_TOKEN_LIMIT)
        }

token_tracker = UserTokenTracker()

# ==============================================================================
# 2. MULTI-PROVIDER AI ADAPTER & MODEL SELECTION
# ==============================================================================
def detect_provider(key):
    """Auto-detect provider from official API key signatures."""
    k = (key or "").strip()
    if k.startswith("gsk_"):
        return "groq"
    if k.startswith("AIza"):
        return "gemini"
    if k.startswith("sk-ant-"):
        return "anthropic"
    if k.startswith("sk-or-"):
        return "openrouter"
    if k.startswith("sk-"):
        return "openai"
    return "unknown"

def hash_key_preview(key):
    """Secure SHA-256 fingerprint preview (never displays full secret)."""
    if not key:
        return ""
    k = key.strip()
    sha = hashlib.sha256(k.encode()).hexdigest()[:8]
    prefix = k[:4]
    suffix = k[-4:] if len(k) > 8 else ""
    return f"{prefix}••••••••{suffix} ({sha})"

def resolve_model_and_endpoint(provider_hint, custom_key, mode, is_advanced, custom_model=""):
    """Determine target domain, model, and authentication parameters without forcing models."""
    key = (custom_key or PRIMARY_API_KEY or "").strip()
    if not key:
        return "fallback", "openai-fast", "", ""

    p = provider_hint.lower() if provider_hint and provider_hint != "auto" else detect_provider(key)
    if p == "unknown":
        p = "groq" if key.startswith("gsk_") else "gemini" if key.startswith("AIza") else "openai"

    # If the user typed or chose an exact custom model, honor it directly!
    selected_model = (custom_model or "").strip()

    if p == "gemini":
        model = selected_model if selected_model else ("gemini-1.5-pro" if (mode == "precision" or is_advanced) else "gemini-1.5-flash")
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    elif p == "openai":
        model = selected_model if selected_model else ("gpt-4o" if (mode == "precision" or is_advanced) else "gpt-4o-mini")
        url = "https://api.openai.com/v1/chat/completions"
    elif p == "anthropic":
        model = selected_model if selected_model else ("claude-3-5-sonnet-20241022" if (mode == "precision" or is_advanced) else "claude-3-5-haiku-20241022")
        url = "https://api.anthropic.com/v1/messages"
    elif p == "openrouter":
        model = selected_model if selected_model else ("meta-llama/llama-3.3-70b-instruct" if (mode == "precision" or is_advanced) else "meta-llama/llama-3.1-8b-instruct:free")
        url = "https://openrouter.ai/api/v1/chat/completions"
    elif p == "groq":
        model = selected_model if selected_model else ("llama-3.3-70b-versatile" if (mode == "precision" or is_advanced) else "llama-3.1-8b-instant")
        url = "https://api.groq.com/openai/v1/chat/completions"
    else:
        return "fallback", "openai-fast", "", ""

    return p, model, key, url

def is_advanced_query(text):
    """Detect if the query asks for advanced reasoning, code, math, or architectural design."""
    patterns = [
        r'\b(code|python|javascript|typescript|c\+\+|java|rust|golang|sql|html|css|bash|regex)\b',
        r'\b(algorithm|architecture|design pattern|refactor|debug|fix|function|class|method|struct)\b',
        r'\b(math|calculus|algebra|formula|proof|equation|probability|statistics|matrix|vector)\b',
        r'\b(explain why|compare|trade-offs|step-by-step|analysis|deep dive|benchmark|system design)\b',
        r'\b(optimize|database|backend|security|auth|jwt|oauth|microservice|concurrency|async)\b'
    ]
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)

def build_system_prompt(is_advanced=False, search_context=""):
    """Produce clean, structured, high-tier responses without filler."""
    if is_advanced:
        prompt = (
            "You are Nova, an elite high-efficiency AI specializing in advanced technical problem-solving.\n"
            "Style Guidelines:\n"
            "1. Zero Preamble: Never begin with filler (e.g. 'Sure!', 'I can help with that', 'Certainly'). Start directly with the answer.\n"
            "2. Clear Structure: Organize with bold headings, concise bullet points, and markdown tables for comparisons.\n"
            "3. Production-Grade Code: When writing code, provide clean, idiomatic, fully functional implementations with explicit language tags (e.g. ```python) and brief explanatory comments.\n"
            "4. Technical Rigor: State time/space complexity ($O(n)$) and architectural trade-offs where relevant."
        )
    else:
        prompt = (
            "You are Nova, an ultra-fast, intelligent AI assistant.\n"
            "Style Guidelines:\n"
            "1. Instant & Direct: Never use filler introductions. Start directly with the answer.\n"
            "2. Structured & Scannable: Use concise bullet points, bold key terms, and short paragraphs for fast reading.\n"
            "3. Accuracy & Clarity: Provide factual, helpful, and easily digestible answers."
        )

    if search_context:
        prompt += f"\n\nLive Verified Web Information:\n{search_context}"
    return prompt

# Fast web search helper (tight 2.0s timeout to never delay chat answers)
def search_web_fast(query, max_results=3):
    try:
        res = http_session.post(
            'https://html.duckduckgo.com/html/',
            data={'q': query},
            timeout=2.0
        )
        page = res.text
        snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', page)
        clean_results = []
        for s in snippets:
            clean = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
            if clean and clean not in clean_results:
                clean_results.append(clean)
            if len(clean_results) >= max_results:
                break
        return clean_results
    except Exception as e:
        print(f"Fast search note: {e}")
        return []

def clean_sse_data(raw_text):
    """Clean internal stream format into pure readable text."""
    if "data: " not in raw_text:
        return raw_text.strip()
    parts = []
    for line in raw_text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("data: ") and "[DONE]" not in trimmed:
            try:
                parsed = json.loads(trimmed[6:])
                choices = parsed.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if not content:
                        content = choices[0].get("text", "")
                    if content:
                        parts.append(content)
            except Exception:
                pass
    return "".join(parts).strip() if parts else raw_text.strip()

# High availability fallback engine (reusing TLS connection for ~0.4s response time)
def call_fallback_sync(messages):
    try:
        payload = {"messages": messages, "model": "openai-fast", "stream": False}
        res = http_session.post(
            "https://text.pollinations.ai/",
            json=payload,
            timeout=18
        )
        cleaned = clean_sse_data(res.text)
        lowered = cleaned.lower()
        if any(term in lowered for term in ["free limit", "pollinations", "pollen", "per ip", "unlimited replies"]):
            return "Nova AI is experiencing peak traffic. Please retry in a brief moment, or connect your own API key in Speed Settings for uninterrupted access."
        return cleaned if cleaned else "I am ready. How can I assist you?"
    except Exception as e:
        print(f"Fast fallback note: {e}")
        return "Nova AI is currently processing high traffic. Please retry your message in a moment."

# ==============================================================================
# 3. DIRECT OUTBOUND STREAMING & SYNC CALLS
# ==============================================================================
def stream_multi_provider(provider, model, key, url, messages, system_prompt, token_budget, temperature):
    """Stream directly from provider's official HTTPS domain."""
    if provider in ["groq", "gemini", "openai", "openrouter"]:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://nova-ai.local"
            headers["X-Title"] = "Nova AI"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": token_budget,
            "temperature": temperature,
            "stream": True
        }

        try:
            res = http_session.post(url, headers=headers, json=payload, stream=True, timeout=28)
            if res.status_code != 200:
                err_msg = f"Provider returned code {res.status_code}"
                try:
                    err_data = res.json()
                    if "error" in err_data:
                        err_msg = err_data["error"].get("message", err_msg)
                except Exception:
                    pass
                yield f"data: {json.dumps({'content': f'Notice: {err_msg}'})}\n\n"
                return

            for line in res.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded == "data: [DONE]":
                    break
                if decoded.startswith("data: "):
                    try:
                        chunk = json.loads(decoded[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield f"data: {json.dumps({'content': delta})}\n\n"
                    except Exception:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Connection note: {str(e)}'})}\n\n"

    elif provider == "anthropic":
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        user_assistant_messages = [m for m in messages if m.get("role") in ["user", "assistant"]]
        payload = {
            "model": model,
            "max_tokens": token_budget,
            "system": system_prompt,
            "messages": user_assistant_messages,
            "temperature": temperature,
            "stream": True
        }
        try:
            res = http_session.post(url, headers=headers, json=payload, stream=True, timeout=28)
            if res.status_code != 200:
                err_msg = f"Anthropic notice: {res.status_code}"
                try:
                    err_data = res.json()
                    if "error" in err_data:
                        err_msg = err_data["error"].get("message", err_msg)
                except Exception:
                    pass
                yield f"data: {json.dumps({'content': f'Notice: {err_msg}'})}\n\n"
                return

            for line in res.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: "):
                    try:
                        event_data = json.loads(decoded[6:])
                        if event_data.get("type") == "content_block_delta":
                            text = event_data.get("delta", {}).get("text", "")
                            if text:
                                yield f"data: {json.dumps({'content': text})}\n\n"
                    except Exception:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Connection note: {str(e)}'})}\n\n"

    else:
        # Fallback pooled generator
        full_reply = call_fallback_sync(messages)
        chunks = re.findall(r'\S+|\s+', full_reply)
        for c in chunks:
            yield f"data: {json.dumps({'content': c})}\n\n"

def sync_multi_provider(provider, model, key, url, messages, system_prompt, token_budget, temperature):
    """Synchronous execution for /chat."""
    if provider in ["groq", "gemini", "openai", "openrouter"]:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://nova-ai.local"
            headers["X-Title"] = "Nova AI"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": token_budget,
            "temperature": temperature,
            "stream": False
        }
        res = http_session.post(url, headers=headers, json=payload, timeout=25)
        if res.status_code == 200:
            data = res.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Provider notice ({res.status_code})")
    elif provider == "anthropic":
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        user_assistant_messages = [m for m in messages if m.get("role") in ["user", "assistant"]]
        payload = {
            "model": model,
            "max_tokens": token_budget,
            "system": system_prompt,
            "messages": user_assistant_messages,
            "temperature": temperature
        }
        res = http_session.post(url, headers=headers, json=payload, timeout=25)
        if res.status_code == 200:
            data = res.json()
            return data["content"][0]["text"]
        else:
            raise Exception(f"Anthropic notice ({res.status_code})")
    else:
        return call_fallback_sync(messages)

# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "engine": "Nova Multi-Provider Neural Engine v4.0",
        "supported_providers": ["gemini", "groq", "openai", "anthropic", "openrouter", "cloud-fallback"],
        "default_models": {
            "gemini": {"turbo": "gemini-1.5-flash", "precision": "gemini-1.5-pro"},
            "groq": {"turbo": "llama-3.1-8b-instant", "precision": "llama-3.3-70b-versatile"},
            "openai": {"turbo": "gpt-4o-mini", "precision": "gpt-4o"},
            "anthropic": {"turbo": "claude-3-5-haiku", "precision": "claude-3-5-sonnet"},
            "openrouter": {"turbo": "meta-llama/llama-3.1-8b-instruct:free", "precision": "meta-llama/llama-3.3-70b-instruct"}
        },
        "token_reset_cycle_hours": 7,
        "free_tier_token_limit": FREE_TIER_TOKEN_LIMIT
    })

@app.route("/api/quota", methods=["GET"])
def get_quota():
    """Retrieve active token usage & 7-hour reset timer for the calling user IP."""
    client_ip = token_tracker.get_client_ip(request)
    custom_key = (request.headers.get("X-API-Key") or request.args.get("api_key") or "").strip()
    has_personal_key = bool(custom_key)
    status = token_tracker.get_status(client_ip, has_personal_key=has_personal_key)
    return jsonify(status)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    mode = data.get("mode", "turbo")
    web_search = data.get("web_search", False)
    custom_key = data.get("api_key", "").strip()
    custom_model = data.get("custom_model") or data.get("model") or ""
    provider_hint = data.get("provider", "auto")

    if not user_message:
        return jsonify({"reply": "Please enter a message!"}), 400

    client_ip = token_tracker.get_client_ip(request)
    # Default users rely on shared Groq engine and ARE strictly capped to 20,000 tokens/7h
    has_personal_key = bool(custom_key)

    # 1. Enforce 7-hour Token Quota
    quota = token_tracker.get_status(client_ip, has_personal_key=has_personal_key)
    if not quota["can_proceed"]:
        resets_in = quota["resets_formatted"]
        limit_msg = (
            f"You have reached your 7-hour free quota ({FREE_TIER_TOKEN_LIMIT:,} tokens) on our shared Groq engine. "
            f"Your limit resets in {resets_in}. "
            "To continue without limit, connect your personal API key (Gemini, Groq, OpenAI, Claude) in Settings."
        )
        return jsonify({"reply": limit_msg, "quota_exceeded": True}), 429

    # 2. Web search context
    search_context = ""
    if web_search:
        results = search_web_fast(user_message)
        if results:
            search_context = "\n[Verified Web Data]:\n" + "\n".join(f"- {r}" for r in results)

    is_advanced = is_advanced_query(user_message) or (mode == "precision")
    system_prompt = build_system_prompt(is_advanced=is_advanced, search_context=search_context)

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        content = msg.get("content") or msg.get("text")
        if msg.get("role") in ["user", "assistant", "bot"] and content:
            role = "assistant" if msg.get("role") in ["assistant", "bot"] else "user"
            formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": user_message})

    provider, model, key, url = resolve_model_and_endpoint(provider_hint, custom_key, mode, is_advanced, custom_model=custom_model)
    token_budget = 1200 if is_advanced else 750
    temperature = 0.4 if is_advanced else 0.6

    try:
        reply = sync_multi_provider(provider, model, key, url, formatted_messages, system_prompt, token_budget, temperature)
        # Track token consumption (approx 1 token per 4 characters)
        estimated_tokens = max(1, (len(user_message) + len(reply)) // 4)
        token_tracker.consume_tokens(client_ip, estimated_tokens)
        ui_mode = f"{provider.capitalize()} ({model}) · {'Precision' if is_advanced else 'Turbo'}"
        return jsonify({"reply": reply, "mode": ui_mode, "searched": bool(search_context)})
    except Exception as e:
        print(f"Chat error: {e}")
        # Graceful fallback
        fallback_reply = call_fallback_sync(formatted_messages)
        token_tracker.consume_tokens(client_ip, max(1, len(fallback_reply) // 4))
        return jsonify({"reply": fallback_reply, "mode": "Cloud Fallback", "searched": bool(search_context)})

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    mode = data.get("mode", "turbo")
    web_search = data.get("web_search", False)
    custom_key = data.get("api_key", "").strip()
    custom_model = data.get("custom_model") or data.get("model") or ""
    provider_hint = data.get("provider", "auto")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    client_ip = token_tracker.get_client_ip(request)
    has_personal_key = bool(custom_key)

    # 1. Enforce 7-hour Token Quota
    quota = token_tracker.get_status(client_ip, has_personal_key=has_personal_key)
    if not quota["can_proceed"]:
        resets_in = quota["resets_formatted"]
        limit_msg = (
            f"You have reached your 7-hour free quota ({FREE_TIER_TOKEN_LIMIT:,} tokens) on our shared Groq engine. "
            f"Your limit resets in {resets_in}. "
            "To continue without limit, connect your personal API key (Gemini, Groq, OpenAI, Claude) in Settings."
        )
        def quota_err_gen():
            yield f"data: {json.dumps({'content': limit_msg})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(quota_err_gen()), mimetype="text/event-stream")

    search_context = ""
    if web_search:
        results = search_web_fast(user_message)
        if results:
            search_context = "\n[Verified Web Data]:\n" + "\n".join(f"- {r}" for r in results)

    is_advanced = is_advanced_query(user_message) or (mode == "precision")
    system_prompt = build_system_prompt(is_advanced=is_advanced, search_context=search_context)

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        content = msg.get("content") or msg.get("text")
        if msg.get("role") in ["user", "assistant", "bot"] and content:
            role = "assistant" if msg.get("role") in ["assistant", "bot"] else "user"
            formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": user_message})

    provider, model, key, url = resolve_model_and_endpoint(provider_hint, custom_key, mode, is_advanced, custom_model=custom_model)
    token_budget = 1200 if is_advanced else 750
    temperature = 0.4 if is_advanced else 0.6

    def generate():
        if search_context:
            yield f"data: {json.dumps({'meta': 'Browsing live web sources...'})}\n\n"

        tokens_generated = 0
        gen = stream_multi_provider(provider, model, key, url, formatted_messages, system_prompt, token_budget, temperature)
        for chunk_event in gen:
            tokens_generated += 1
            yield chunk_event

        yield "data: [DONE]\n\n"
        # Consume tokens on completion
        total_tokens = max(1, (len(user_message) // 4) + tokens_generated)
        token_tracker.consume_tokens(client_ip, total_tokens)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
