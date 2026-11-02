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

# Initialize Groq client if key is available
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
groq_client = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize Groq client: {e}")

# Web search helper using DuckDuckGo HTML (no API key required, zero external dependencies)
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
        print(f"Search error: {e}")
        return []

def should_search_web(user_message, web_search_forced=False):
    if web_search_forced:
        return True
    triggers = [
        "today", "yesterday", "latest", "recent", "news", "current", "weather",
        "score", "stock", "price", "update", "2024", "2025", "2026", "who is the current",
        "what happened", "release date", "election", "match"
    ]
    msg = user_message.lower()
    return any(t in msg for t in triggers)

def clean_sse_data(raw_text):
    """If text contains SSE data: chunks, extract the clean inner content."""
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

# Free fallback API non-streaming (using anonymous openai-fast tier)
def call_free_fallback_sync(messages):
    try:
        payload = json.dumps({"messages": messages, "model": "openai-fast", "stream": False}).encode('utf-8')
        req = urllib.request.Request(
            "https://text.pollinations.ai/",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode('utf-8', errors='ignore')
            return clean_sse_data(raw)
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

# Free fallback API streaming (using anonymous openai-fast tier)
def call_free_fallback_stream(messages):
    try:
        payload = json.dumps({"messages": messages, "model": "openai-fast", "stream": True}).encode('utf-8')
        req = urllib.request.Request(
            "https://text.pollinations.ai/",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=40) as response:
            for raw_line in response:
                line = raw_line.decode('utf-8', errors='ignore')
                if line.startswith("data: "):
                    if "[DONE]" in line:
                        break
                    try:
                        data = json.loads(line[6:].strip())
                        choices = data.get("choices", [])
                        if choices:
                            chunk = choices[0].get("delta", {}).get("content", "")
                            if chunk:
                                yield chunk
                    except Exception:
                        pass
                else:
                    if line.strip():
                        yield line
    except Exception as e:
        yield f"Error in stream: {str(e)}"

@app.route("/")
def index():
    has_groq = bool(GROQ_API_KEY)
    return render_template("index.html", has_groq=has_groq)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "has_groq_key": bool(GROQ_API_KEY),
        "models_available": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "free-cloud-ai"]
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    model_choice = data.get("model", "llama-3.1-8b-instant")
    web_search = data.get("web_search", False)

    if not user_message:
        return jsonify({"reply": "Please enter a message!"}), 400

    search_context = ""
    if should_search_web(user_message, web_search):
        search_results = search_web(user_message)
        if search_results:
            search_context = "\n\n[Live Real-World Web Search Results]:\n" + "\n".join(f"- {r}" for r in search_results)

    system_prompt = (
        "You are an intelligent, helpful, and friendly AI assistant. "
        "Answer questions clearly, concisely, and accurately. "
        "Format code snippets with standard markdown code blocks (```language). "
    )
    if search_context:
        system_prompt += (
            f"\nYou have access to real-time web search information for up-to-date data:{search_context}\n"
            "Use this live information to answer accurately and reference current facts."
        )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        if msg.get("role") in ["user", "assistant", "bot"] and (msg.get("content") or msg.get("text")):
            content = msg.get("content") or msg.get("text")
            role = "assistant" if msg.get("role") in ["assistant", "bot"] else "user"
            formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": user_message})

    # 1. Try Groq if client is configured
    if groq_client:
        try:
            groq_model = "llama-3.1-8b-instant" if "8b" in model_choice else "llama-3.3-70b-versatile"
            response = groq_client.chat.completions.create(
                model=groq_model,
                messages=formatted_messages,
                max_tokens=1500,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            return jsonify({"reply": reply, "provider": f"Groq ({groq_model})", "searched": bool(search_context)})
        except Exception as e:
            print(f"Groq API error, falling back to free gateway: {e}")

    # 2. Fallback to free unlimited cloud provider
    try:
        reply = call_free_fallback_sync(formatted_messages)
        if not reply:
            reply = "I received an empty response. Please try asking again."
        return jsonify({"reply": reply, "provider": "Free Cloud AI", "searched": bool(search_context)})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    model_choice = data.get("model", "llama-3.1-8b-instant")
    web_search = data.get("web_search", False)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    search_context = ""
    if should_search_web(user_message, web_search):
        search_results = search_web(user_message)
        if search_results:
            search_context = "\n\n[Live Real-World Web Search Results]:\n" + "\n".join(f"- {r}" for r in search_results)

    system_prompt = (
        "You are an intelligent, helpful, and friendly AI assistant. "
        "Answer questions clearly, concisely, and accurately. "
        "Format code snippets with standard markdown code blocks (```language). "
    )
    if search_context:
        system_prompt += (
            f"\nYou have access to real-time web search information for up-to-date data:{search_context}\n"
            "Use this live information to answer accurately and reference current facts."
        )

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        if msg.get("role") in ["user", "assistant", "bot"] and (msg.get("content") or msg.get("text")):
            content = msg.get("content") or msg.get("text")
            role = "assistant" if msg.get("role") in ["assistant", "bot"] else "user"
            formatted_messages.append({"role": role, "content": content})
    formatted_messages.append({"role": "user", "content": user_message})

    def generate():
        if search_context:
            yield f"data: {json.dumps({'meta': 'Searching web for real-world updates...'})}\n\n"

        # Try Groq streaming first
        if groq_client:
            try:
                groq_model = "llama-3.1-8b-instant" if "8b" in model_choice else "llama-3.3-70b-versatile"
                stream = groq_client.chat.completions.create(
                    model=groq_model,
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
                print(f"Groq stream error, falling back to free gateway: {e}")

        # Fallback to free gateway streaming
        for chunk in call_free_fallback_stream(formatted_messages):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
