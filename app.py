import os
import json
import re
import html
import urllib.request
import urllib.parse
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize AI engine (accepts NOVA_API_KEY or GROQ_API_KEY for seamless setup)
PRIMARY_API_KEY = (os.environ.get("NOVA_API_KEY") or os.environ.get("GROQ_API_KEY") or "").strip()
engine_client = None

if PRIMARY_API_KEY:
    try:
        from groq import Groq
        engine_client = Groq(api_key=PRIMARY_API_KEY)
        print("Nova Neural Engine initialized with primary accelerator.")
    except Exception as e:
        print(f"Primary accelerator initialization notice: {e}")

# Web search helper using DuckDuckGo HTML (zero dependencies, completely private)
def search_web(query, max_results=4):
    try:
        data = urllib.parse.urlencode({'q': query}).encode('utf-8')
        req = urllib.request.Request(
            'https://html.duckduckgo.com/html/',
            data=data,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            page = response.read().decode('utf-8', errors='ignore')

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
        print(f"Web search note: {e}")
        return []

def should_search_web(user_message, web_search_forced=False):
    if web_search_forced:
        return True
    triggers = [
        "today", "yesterday", "latest", "recent", "news", "current", "weather",
        "score", "stock", "price", "update", "2024", "2025", "2026", "who is the current",
        "what happened", "release date", "election", "match", "trending"
    ]
    msg = user_message.lower()
    return any(t in msg for t in triggers)

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

# High availability fallback engine (completely white-labeled, never exposes raw errors)
def call_fallback_sync(messages):
    try:
        payload = json.dumps({"messages": messages, "model": "openai-fast", "stream": False}).encode('utf-8')
        req = urllib.request.Request(
            "https://text.pollinations.ai/",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read().decode('utf-8', errors='ignore')
            cleaned = clean_sse_data(raw)
            return cleaned if cleaned else "I am ready. How can I assist you?"
    except Exception as e:
        print(f"Fallback engine note: {e}")
        return "Nova AI is currently processing high traffic. Please retry your message in a moment."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "engine": "Nova Neural Engine v2.5",
        "modes": ["turbo", "precision"],
        "web_search": "active"
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    mode = data.get("mode", "turbo")
    web_search = data.get("web_search", False)

    if not user_message:
        return jsonify({"reply": "Please enter a message!"}), 400

    search_context = ""
    if should_search_web(user_message, web_search):
        search_results = search_web(user_message)
        if search_results:
            search_context = "\n\n[Live Real-World Web Knowledge]:\n" + "\n".join(f"- {r}" for r in search_results)

    system_prompt = (
        "You are Nova, an advanced, highly intelligent, and helpful AI assistant. "
        "Answer questions clearly, concisely, and accurately. "
        "When providing code, always specify the language in markdown blocks (```language). "
    )
    if search_context:
        system_prompt += (
            f"\nYou have access to real-time verified web information:{search_context}\n"
            "Use this live information to provide up-to-date and factual answers."
        )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        content = msg.get("content") or msg.get("text")
        if msg.get("role") in ["user", "assistant", "bot"] and content:
            role = "assistant" if msg.get("role") in ["assistant", "bot"] else "user"
            formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": user_message})

    ui_mode = "Turbo Mode" if mode == "turbo" else "Precision Mode"

    # Primary engine execution
    if engine_client:
        try:
            internal_model = "llama-3.1-8b-instant" if mode == "turbo" else "llama-3.3-70b-versatile"
            response = engine_client.chat.completions.create(
                model=internal_model,
                messages=formatted_messages,
                max_tokens=1500,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            return jsonify({"reply": reply, "mode": ui_mode, "searched": bool(search_context)})
        except Exception as e:
            print(f"Primary engine note: {e}")

    # Seamless fallback engine
    try:
        reply = call_fallback_sync(formatted_messages)
        return jsonify({"reply": reply, "mode": ui_mode, "searched": bool(search_context)})
    except Exception as e:
        print(f"Chat route error: {e}")
        return jsonify({"reply": "Nova AI is currently busy. Please retry in a moment."}), 500

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    mode = data.get("mode", "turbo")
    web_search = data.get("web_search", False)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    search_context = ""
    if should_search_web(user_message, web_search):
        search_results = search_web(user_message)
        if search_results:
            search_context = "\n\n[Live Real-World Web Knowledge]:\n" + "\n".join(f"- {r}" for r in search_results)

    system_prompt = (
        "You are Nova, an advanced, highly intelligent, and helpful AI assistant. "
        "Answer questions clearly, concisely, and accurately. "
        "When providing code, always specify the language in markdown blocks (```language). "
    )
    if search_context:
        system_prompt += (
            f"\nYou have access to real-time verified web information:{search_context}\n"
            "Use this live information to provide up-to-date and factual answers."
        )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        content = msg.get("content") or msg.get("text")
        if msg.get("role") in ["user", "assistant", "bot"] and content:
            role = "assistant" if msg.get("role") in ["assistant", "bot"] else "user"
            formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": user_message})

    def generate():
        if search_context:
            yield f"data: {json.dumps({'meta': 'Browsing live web sources...'})}\n\n"

        if engine_client:
            try:
                internal_model = "llama-3.1-8b-instant" if mode == "turbo" else "llama-3.3-70b-versatile"
                stream = engine_client.chat.completions.create(
                    model=internal_model,
                    messages=formatted_messages,
                    max_tokens=1500,
                    temperature=0.7,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield f"data: {json.dumps({'content': delta})}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                print(f"Primary stream accelerator note: {e}")

        # Ultra-resilient token stream generator
        full_reply = call_fallback_sync(formatted_messages)
        chunks = re.findall(r'\S+|\s+', full_reply)
        for c in chunks:
            yield f"data: {json.dumps({'content': c})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
