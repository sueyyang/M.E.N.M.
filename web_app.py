#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAKE ENGLISH NOT MUSIC — Harsh Noise Web Version (no fluidsynth/ffmpeg)
- Pure Python + NumPy audio synthesis.
- Text -> bits -> D-beat–like pattern driving noise-based kick/snare/hat/crash.
- Audio is generated as mono 16-bit WAV in-memory and played via <audio> in browser.
"""

import os
import mimetypes
import base64
import io
import wave

import numpy as np
from flask import Flask, request, render_template_string, send_file, abort

# ------------ Config ------------

APP_TITLE = "MAKE ENGLISH NOT MUSIC (NOISE WEB)"
BPM = 300
SUBDIV = 16                # 16th-note grid
ADD_DBEAT_INTRO = True
ADD_DBEAT_OUTRO = True
ALT_KICK_SNARE = True
GHOST_HAT_ON_REST = True   # 为噪音风格，rest 也打一丢丢高频噪声

SR = 44100                 # sample rate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 背景图片候选
BG_CANDIDATES = [
    "background.png", "background.jpg", "background.jpeg",
    "background.gif", "background.bmp", "background.webp",
    "未标题-16.png"
]
BG_PATH = None
for name in BG_CANDIDATES:
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        BG_PATH = p
        break

app = Flask(__name__)

# ------------ Bit logic ------------

def text_to_bits(s: str) -> str:
    data = s.encode("ascii", "ignore")
    return "".join(f"{b:08b}" for b in data)

# ------------ Harsh Noise Drum Synth ------------

def make_kick(sr: int, dur: float) -> np.ndarray:
    """Harsh, distorted low kick."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Pitch sweep: high to low
    f0, f1 = 140.0, 40.0
    sweep = f0 * np.exp(-6 * t) + f1
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    tone = np.sin(phase)
    env = np.exp(-8 * t)
    noise = np.random.randn(len(t)) * 0.35
    x = tone * env * 1.2 + noise * env * 0.7
    return np.tanh(4.0 * x).astype(np.float32)

def make_snare(sr: int, dur: float) -> np.ndarray:
    """Noisy snare: broadband noise with fast decay."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t))
    env = np.exp(-18 * t)
    # slight band emphasis (~2k–4k)
    band = np.convolve(noise, np.array([1, -1, 1, -1]), mode="same")
    x = (noise * 0.4 + band * 0.6) * env
    return np.tanh(3.5 * x).astype(np.float32)

def make_hat(sr: int, dur: float, soft: bool = False) -> np.ndarray:
    """High-frequency metallic-ish hat."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t))
    # crude high-pass: diff
    hp = np.convolve(noise, np.array([1, -1]), mode="same")
    env = np.exp(-45 * t)
    amp = 0.25 if soft else 0.4
    x = hp * env * amp
    return np.tanh(3.0 * x).astype(np.float32)

def make_crash(sr: int, dur: float) -> np.ndarray:
    """Longer noisy crash / wash."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t))
    env = np.exp(-2.5 * t)
    band = np.convolve(noise, np.array([1, -1, 1, -1, 1, -1]), mode="same")
    x = (noise * 0.3 + band * 0.7) * env
    return np.tanh(3.0 * x).astype(np.float32)

def add_hit(buf: np.ndarray, start_idx: int, hit: np.ndarray):
    """Add one hit waveform into main buffer with simple mix."""
    if start_idx >= len(buf):
        return
    end_idx = min(len(buf), start_idx + len(hit))
    length = end_idx - start_idx
    if length <= 0:
        return
    buf[start_idx:end_idx] += hit[:length]

def text_to_noise_wav(text: str):
    """Text -> bitstring -> harsh noise D-beat pattern -> mono WAV bytes."""

    bits = text_to_bits(text if text else "make english not music")

    # 16th-note step duration
    sec_per_quarter = 60.0 / BPM
    step_dur = sec_per_quarter / 4.0

    # Rough intro/outro sizes in "steps"
    intro_steps = 16 if ADD_DBEAT_INTRO else 0   # 1 bar of 4/4 at 16th grid
    outro_steps = 16 if ADD_DBEAT_OUTRO else 0
    total_steps = intro_steps + len(bits) + outro_steps

    total_dur = total_steps * step_dur + 0.5  # extra tail
    n_samples = int(total_dur * SR) + 1
    audio = np.zeros(n_samples, dtype=np.float32)

    # Prebuild hits (brutal, noisy)
    hit_dur = step_dur * 0.9
    kick = make_kick(SR, hit_dur)
    snare = make_snare(SR, hit_dur)
    hat = make_hat(SR, hit_dur, soft=False)
    hat_ghost = make_hat(SR, hit_dur * 0.6, soft=True)
    crash = make_crash(SR, step_dur * 8.0)

    step_samples = int(step_dur * SR)

    # --- Intro: simple D-beat-ish bar ---
    pos = 0
    if ADD_DBEAT_INTRO:
        pattern = [kick, snare, kick, snare]
        spacing = step_samples * 4  # quarter-note spacing at this BPM
        for h in pattern:
            add_hit(audio, pos, h)
            add_hit(audio, pos, hat)
            pos += spacing

    # --- Main bits ---
    toggle = True
    for b in bits:
        if b == "1":
            drum = kick if (toggle or not ALT_KICK_SNARE) else snare
            toggle = not toggle if ALT_KICK_SNARE else toggle
            add_hit(audio, pos, drum)
            add_hit(audio, pos, hat)
        else:
            if GHOST_HAT_ON_REST:
                add_hit(audio, pos, hat_ghost)
        pos += step_samples

    # --- Outro ---
    if ADD_DBEAT_OUTRO:
        pattern = [kick, snare, kick, snare]
        spacing = step_samples * 4
        for h in pattern:
            add_hit(audio, pos, h)
            add_hit(audio, pos, hat)
            pos += spacing
        add_hit(audio, pos, crash)

    # Normalize
    max_val = float(np.max(np.abs(audio)))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95

    # Convert to 16-bit PCM
    pcm = (audio * 32767.0).astype(np.int16)

    # Write WAV to memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    wav_bytes = buf.getvalue()

    # Approx duration for UI
    approx_dur = len(pcm) / SR
    return wav_bytes, bits, approx_dur

# ------------ HTML Templates ------------

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #000;
      color: #fff;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Mono", monospace;
    }
    .wrap {
      display: flex;
      min-height: 100vh;
      flex-direction: row;
    }
    .left {
      flex: 1.2;
      background: #000;
      position: relative;
      overflow: hidden;
      min-height: 260px;
    }
    .left-inner {
      position: absolute;
      inset: 0;
      background: #000;
      {% if has_bg %}
      background-image: url("/bg");
      background-repeat: no-repeat;
      background-position: center;
      background-size: contain;
      {% else %}
      display:flex;
      align-items:center;
      justify-content:center;
      color:#666;
      {% endif %}
    }
    .right {
      width: 360px;
      max-width: 100%%;
      padding: 20px 18px 32px;
      background: #000;
      border-left: 1px solid #222;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 16px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    label {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #aaa;
    }
    input[type=text] {
      width: 100%%;
      padding: 8px 10px;
      margin-top: 6px;
      margin-bottom: 14px;
      border: 1px solid #444;
      background: #080808;
      color: #fff;
      font-family: inherit;
      font-size: 14px;
    }
    button {
      padding: 8px 16px;
      border: 1px solid #666;
      background: #222;
      color: #fff;
      cursor: pointer;
      font-family: inherit;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
    }
    button:hover { background: #333; }
    .hint {
      margin-top: 16px;
      font-size: 12px;
      color: #777;
      line-height: 1.5;
    }
    @media (max-width: 768px) {
      .wrap { flex-direction: column; }
      .left {
        height: 35vh;
        min-height: 180px;
      }
      .right {
        width: 100%%;
        border-left: none;
        border-top: 1px solid #222;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <div class="left-inner">
        {% if not has_bg %}
        <div>NO BACKGROUND IMAGE FOUND</div>
        {% endif %}
      </div>
    </div>
    <div class="right">
      <h1>{{ title }}</h1>
      <form method="post" action="/generate">
        <label for="text">Input English</label><br>
        <input id="text" name="text" type="text"
               value="make english not music" autocomplete="off">
        <button type="submit">Generate</button>
      </form>
      <div class="hint">
        Type an English sentence. It will be encoded into a bit sequence
        and turned into a harsh D-beat / noise drum pattern.<br>
        After you submit, you'll see a result page where you can listen
        to the generated sound directly in the browser.
      </div>
    </div>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ title }} — Result</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      margin: 0;
      background: #000;
      color: #fff;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Mono", monospace;
      padding: 20px;
    }
    a { color: #aaa; text-decoration: none; }
    a:hover { color: #fff; }
    h1 {
      font-size: 20px;
      margin: 0 0 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .meta {
      font-size: 13px;
      color: #bbb;
      margin-bottom: 14px;
      line-height: 1.6;
    }
    .text-block {
      padding: 10px 12px;
      border: 1px solid #333;
      background: #060606;
      font-size: 13px;
      margin-bottom: 16px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    audio {
      width: 100%%;
      margin-top: 8px;
    }
    .back {
      margin-top: 18px;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="meta">
    Bits: {{ bits }} &nbsp;·&nbsp; Estimated duration ≈ {{ duration }} s
  </div>
  <div class="text-block">
    {{ text }}
  </div>
  <audio controls>
    <source src="data:audio/wav;base64,{{ b64 }}" type="audio/wav">
    Your browser does not support the audio element.
  </audio>
  <div class="back">
    <a href="/">&#8592; Back</a>
  </div>
</body>
</html>
"""

# ------------ Routes ------------

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        INDEX_HTML,
        title=APP_TITLE,
        has_bg=(BG_PATH is not None),
    )

@app.route("/bg")
def bg():
    if not BG_PATH:
        abort(404)
    mime, _ = mimetypes.guess_type(BG_PATH)
    return send_file(BG_PATH, mimetype=mime or "image/png")

@app.route("/generate", methods=["POST"])
def generate():
    text = request.form.get("text", "").strip()
    if not text:
        text = "make english not music"

    wav_bytes, bits, dur = text_to_noise_wav(text)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    dur_str = f"{dur:.2f}"

    return render_template_string(
        RESULT_HTML,
        title=APP_TITLE,
        text=text,
        bits=len(bits),
        duration=dur_str,
        b64=b64,
    )

if __name__ == "__main__":
    # 跟之前一样，用 5050 方便你局域网访问
    app.run(host="0.0.0.0", port=5050, debug=False)
