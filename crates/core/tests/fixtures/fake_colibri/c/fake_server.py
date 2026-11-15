# A minimal OpenAI-style server used by the fake `coli start`.
import json, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]); LOAD_SECONDS = float(sys.argv[2]); T0 = time.time()

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.send_header("Connection", "close")
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            if time.time() - T0 < LOAD_SECONDS: return self.send_json(503, {"status": "restarting"})
            return self.send_json(200, {"status": "ok", "arch": "fake", "context_tokens": 8192, "max_tokens": 1024})
        if self.path == "/v1/models": return self.send_json(200, {"data": [{"id": "fake-model"}]})
        if self.path == "/profile": return self.send_json(200, {"seq": 0, "turns": []})
        self.send_json(404, {"error": {"message": "nope"}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(n) or b"{}")
        last = [m for m in body.get("messages", []) if m["role"] == "user"][-1]["content"]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close"); self.end_headers()  # close-framed, like colibri
        def ev(delta, extra=None):
            o = {"choices": [{"index": 0, "delta": delta}]}; o.update(extra or {})
            self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode()); self.wfile.flush()
        try:
            for w in ("Fake answer to: " + last).split(" "):
                ev({"content": w + " "}); time.sleep(0.03)
            ev({}, {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
