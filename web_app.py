import os
import mimetypes
import base64
import io
import wave

import numpy as np
from flask import Flask, request, render_template_string, send_file, abort

APP_TITLE = "MAKE ENGLISH NOT MUSIC (NOISE WEB)"
BPM = 190
SUBDIV = 8
ADD_DBEAT_INTRO = True
ADD_DBEAT_OUTRO = True
ALT_KICK_SNARE = True
GHOST_HAT_ON_REST = True

SR = 44100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BG_CANDIDATES = [
    "background.png",
    "未标题-16.png"
]
BG_PATH = None
for name in BG_CANDIDATES:
    p = os.path.join(BASE_DIR, name)
    if os.path.exists(p):
        BG_PATH = p
        break

app = Flask(__name__)

def text_to_bits(s: str) -> str:
    data = s.encode("ascii", "ignore")
    return "".join(f"{b:08b}" for b in data)

def make_kick(sr: int, dur: float) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    base = 70.0
    sweep = base * (1.0 + 4.0 * np.exp(-10.0 * t))
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    tone = np.sin(phase)
    env = np.exp(-12.0 * t)
    click_len = max(1, int(0.003 * sr))
    click = np.zeros_like(t)
    click[:click_len] = np.linspace(1.0, 0.0, click_len)
    noise = np.random.randn(len(t)) * 0.2
    x = tone * env * 1.4 + noise * env * 0.6 + click * 0.8
    return np.tanh(6.0 * x).astype(np.float32)

def make_snare(sr: int, dur: float) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t))
    env = np.exp(-18.0 * t)
    crack = np.sin(2 * np.pi * 190.0 * t) + 0.5 * np.sin(2 * np.pi * 380.0 * t)
    band = np.convolve(noise, np.array([1.0, -2.0, 3.0, -2.0, 1.0]), mode="same")
    x = (noise * 0.5 + band * 0.8 + crack * 0.7) * env
    return np.tanh(4.0 * x).astype(np.float32)

def make_hat(sr: int, dur: float, soft: bool = False) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t))
    hp = np.convolve(noise, np.array([1.0, -0.6, -0.4]), mode="same")
    env = np.exp(-55.0 * t)
    base_amp = 0.18 if soft else 0.32
    jitter = 0.8 + 0.4 * np.random.rand()
    x = hp * env * base_amp * jitter
    return np.tanh(3.5 * x).astype(np.float32)

def make_crash(sr: int, dur: float) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t))
    env = np.exp(-2.5 * t)
    band = np.convolve(noise, np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), mode="same")
    x = (noise * 0.3 + band * 0.7) * env
    return np.tanh(3.0 * x).astype(np.float32)

def add_hit(buf: np.ndarray, start_idx: int, hit: np.ndarray):
    if start_idx >= len(buf):
        return
    end_idx = min(len(buf), start_idx + len(hit))
    length = end_idx - start_idx
    if length <= 0:
        return
    buf[start_idx:end_idx] += hit[:length]

def text_to_noise_wav(text: str):
    bits = text_to_bits(text if text else "make english not music")

    sec_per_quarter = 60.0 / BPM
    step_dur = sec_per_quarter / 4.0

    intro_steps = 16 if ADD_DBEAT_INTRO else 0
    outro_steps = 16 if ADD_DBEAT_OUTRO else 0
    total_steps = intro_steps + len(bits) + outro_steps

    total_dur = total_steps * step_dur + 0.5
    n_samples = int(total_dur * SR) + 1
    audio = np.zeros(n_samples, dtype=np.float32)

    hit_dur = step_dur * 0.9
    kick = make_kick(SR, hit_dur)
    snare = make_snare(SR, hit_dur)
    hat = make_hat(SR, hit_dur, soft=False)
    hat_ghost = make_hat(SR, hit_dur * 0.6, soft=True)
    crash = make_crash(SR, step_dur * 8.0)
    open_hat = make_hat(SR, hit_dur * 1.2, soft=False)

    step_samples = int(step_dur * SR)

    pos = 0
    if ADD_DBEAT_INTRO:
        pattern = [kick, snare, kick, snare]
        spacing = step_samples * 4
        for h in pattern:
            add_hit(audio, pos, h)
            if np.random.rand() < 0.6:
                add_hit(audio, pos, hat)
            else:
                add_hit(audio, pos, open_hat)
            if np.random.rand() < 0.25:
                add_hit(audio, pos + step_samples, snare)
            pos += spacing

    toggle = True
    for b in bits:
        if b == "1":
            if ALT_KICK_SNARE:
                if toggle:
                    drum = kick
                else:
                    drum = snare
                if np.random.rand() < 0.2:
                    drum = snare if drum is kick else kick
                toggle = not toggle
            else:
                drum = kick
            add_hit(audio, pos, drum)
            if np.random.rand() < 0.7:
                add_hit(audio, pos, hat)
            else:
                add_hit(audio, pos, open_hat)
            if np.random.rand() < 0.18:
                add_hit(audio, pos + step_samples // 2, snare)
            if np.random.rand() < 0.12:
                add_hit(audio, pos + step_samples // 4, hat_ghost)
        else:
            if GHOST_HAT_ON_REST:
                if np.random.rand() < 0.8:
                    add_hit(audio, pos, hat_ghost)
                if np.random.rand() < 0.1:
                    add_hit(audio, pos + step_samples // 2, hat_ghost)
        pos += step_samples

    if ADD_DBEAT_OUTRO:
        pattern = [kick, snare, kick, snare]
        spacing = step_samples * 4
        for h in pattern:
            add_hit(audio, pos, h)
            if np.random.rand() < 0.6:
                add_hit(audio, pos, hat)
            else:
                add_hit(audio, pos, open_hat)
            if np.random.rand() < 0.25:
                add_hit(audio, pos + step_samples, snare)
            pos += spacing
        add_hit(audio, pos, crash)

    max_val = float(np.max(np.abs(audio)))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95

    pcm = (audio * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    wav_bytes = buf.getvalue()

    approx_dur = len(pcm) / SR
    return wav_bytes, bits, approx_dur

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
    app.run(host="0.0.0.0", port=5050, debug=False)
