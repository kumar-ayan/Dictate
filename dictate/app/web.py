from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 90
SAMPLE_RATE = 16_000
WEB_ROOT = Path(__file__).with_name("web")


def _transcribe_payload(payload, asr):
    import numpy as np
    from ..data.audio import decode_audio
    from .pipeline import energy_trim

    try:
        wave, sample_rate = decode_audio(payload)
    except Exception as exc:
        raise ValueError("Could not read that recording. Try recording again.") from exc
    audio = energy_trim(np.asarray(wave, dtype=np.float32), sample_rate=sample_rate)
    seconds = len(audio) / sample_rate
    if seconds > MAX_AUDIO_SECONDS:
        raise ValueError(f"Keep recordings under {MAX_AUDIO_SECONDS} seconds.")
    if seconds < 0.3:
        raise ValueError("Recording was too short or silent. Try speaking a little longer.")
    started = time.perf_counter()
    text = asr(audio)
    elapsed = time.perf_counter() - started
    return {"text": text, "seconds": seconds, "asr_seconds": elapsed}


def _handler(asr, inference_lock):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css", "/mic-processor.js": "mic-processor.js"}
            if self.path not in files:
                self._json(404, {"error": "Page not found."})
                return
            file = WEB_ROOT / files[self.path]
            body = file.read_bytes()
            content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in ("application/javascript",):
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/transcribe":
                self._json(404, {"error": "Endpoint not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "Invalid recording size."})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(413, {"error": "Recording is empty or too large. Keep it under 90 seconds and try again."})
                return
            payload = self.rfile.read(length)
            if len(payload) != length:
                self._json(400, {"error": "Recording upload was incomplete. Try again."})
                return
            try:
                with inference_lock:
                    result = _transcribe_payload(payload, asr)
                self._json(200, result)
            except ValueError as exc:
                self._json(422, {"error": str(exc)})
            except Exception:
                self._json(500, {"error": "Local transcription failed. Check the server window and try again."})

        def log_message(self, format, *args):
            return

    return Handler


def run_web(config, port=8765):
    import torch
    from ..tokenizer import load
    from .run import _load_asr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = load()
    _, asr, _ = _load_asr(device, sp)
    server = ThreadingHTTPServer(("127.0.0.1", int(port)), _handler(asr, threading.Lock()))
    server.daemon_threads = True
    print(f"Local speech tester ready: http://127.0.0.1:{server.server_port}")
    print("Audio is processed in memory and is not written to disk. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Speech tester stopped.")
    finally:
        server.server_close()
