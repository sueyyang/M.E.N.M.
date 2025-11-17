# MAKE ENGLISH NOT MUSIC — Harsh Noise Web Drum Machine

**Input English → D-beat / Harsh Noise / Power Electronics drum patterns.**  
A browser-based text-to-noise generator for research and experimental sound.

### How it works
1. Type any English text.
2. Text is converted into a binary (bit) sequence.
3. Bits drive a noise-based D-beat drum engine (pure Python synthesis).
4. The result is rendered as audio (WAV) and played directly in the browser.

### Stack
- Python + Flask (Web server)
- NumPy (audio DSP, no FluidSynth / no SoundFont / no ffmpeg)
- Pure PCM waveform synthesis for kick/snare/hat/crash

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python web_app.py
