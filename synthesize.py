"""Self-initiated original 20-second ambient instrumental study.

Question: can self-authored synthesis yield a moderate-level, seamless stereo
PCM WAV suitable for further listening assessment? This is not a client score,
professional-quality claim, or an accepted brief. No sampled/borrowed music,
voice, third-party audio or network calls. Icon references: N/A (audio only).
Run: python synthesize.py
NumPy is the only non-stdlib dependency; use the already-installed runtime.
"""
from pathlib import Path
import hashlib
import json
import math
import struct
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent
SR = 48_000
DURATION = 20.0
N = int(SR * DURATION)
t = np.arange(N, dtype=np.float64) / SR
TAU = 2.0 * math.pi


def loop_frequency(hz):
    """Integer cycles per full loop: oscillator phase closes at t=20s."""
    return round(hz * DURATION) / DURATION


def sine(hz, phase=0.0):
    return np.sin(TAU * loop_frequency(hz) * t + phase)


def midi(note):
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


left = np.zeros(N, dtype=np.float64)
right = np.zeros(N, dtype=np.float64)

# Soft D/A/C/F sustained register. Only low, sparse harmonics; no saw/noise.
for i, (note, level) in enumerate(((38, .24), (45, .14), (50, .115),
                                  (53, .085), (60, .038))):
    f = midi(note)
    motion = .82 + .12 * np.sin(TAU * (i + 1) * t / DURATION + i)
    core = sine(f, .4 * i) + .17 * sine(2 * f, .7 + i)
    upper = sine(f + .10, 1.1 + i) + .10 * sine(2 * f - .10, i)
    left += level * motion * (.62 * core + .38 * upper)
    right += level * motion * (.62 * sine(f - .10, .8 + i)
                               + .38 * sine(f + .05, .3 + i))

# Original sparse four-note contour D4/F4/E4/A3; each attack is gradual.
# Circular local time means events/delay tails at the seam remain continuous.
for start, note, length, pan in ((.8, 62, 4.3, -.30), (5.4, 65, 4.6, .32),
                                 (10.5, 64, 4.0, -.15), (15.1, 57, 5.2, .22)):
    local = np.mod(t - start, DURATION)
    active = local < length
    envelope = np.where(active, np.sin(math.pi * np.minimum(local / length, 1)) ** 2,
                        0.0)
    f = midi(note)
    tone = (.88 * sine(f, .6) + .10 * sine(2 * f, .1)
            + .025 * sine(3 * f, .9)) * envelope * .11
    left += tone * math.sqrt((1 - pan) / 2)
    right += tone * math.sqrt((1 + pan) / 2)

# Circular stereo echo: finite, transparent and editable. No external reverb.
dry_l, dry_r = left.copy(), right.copy()
for seconds, gain in ((.173, .13), (.397, .085), (.823, .045), (1.337, .026)):
    shift = round(seconds * SR)
    left += gain * np.roll(dry_r, shift)
    right += gain * np.roll(dry_l, shift + 311)

audio = np.stack((left, right), axis=1)
audio -= audio.mean(axis=0)
audio *= .10 / np.sqrt(np.mean(audio ** 2))  # -20 dBFS RMS, before PCM.
pre_peak = float(np.max(np.abs(audio)))
if pre_peak > .50:
    audio *= .50 / pre_peak
pcm = np.round(audio * 32767).astype('<i2')
wav_path = ROOT / 'ambient-loop.wav'
with wave.open(str(wav_path), 'wb') as out:
    out.setnchannels(2)
    out.setsampwidth(2)
    out.setframerate(SR)
    out.writeframes(pcm.tobytes())

# Reopen actual written PCM; measurements do not substitute for listening.
with wave.open(str(wav_path), 'rb') as check:
    actual = np.frombuffer(check.readframes(check.getnframes()), '<i2')
    actual = actual.reshape(-1, check.getnchannels()).astype(np.float64) / 32768
    params = {'sample_rate_hz': check.getframerate(), 'channels': check.getnchannels(),
              'sample_width_bytes': check.getsampwidth(), 'frames': check.getnframes(),
              'duration_seconds': check.getnframes() / check.getframerate()}
raw = wav_path.read_bytes()
chunks, pos = [], 12
while pos + 8 <= len(raw):
    name, size = struct.unpack_from('<4sI', raw, pos)
    chunks.append({'id': name.decode('ascii'), 'bytes': size})
    pos += 8 + size + (size % 2)
peak = float(np.max(np.abs(actual)))
rms = float(np.sqrt(np.mean(actual ** 2)))
seam = actual[0] - actual[-1]
qa = {
    'type': 'self-initiated original synthesis capability study',
    'main_render_count': 1, 'correction_render_count': 0,
    'format': params, 'riff_chunks': chunks,
    'peak_linear': peak, 'peak_dbfs': 20 * math.log10(peak),
    'rms_linear': rms, 'rms_dbfs': 20 * math.log10(rms),
    'dc_per_channel': actual.mean(axis=0).tolist(),
    'seam_delta_linear_per_channel': seam.tolist(),
    'seam_delta_pcm_lsb_per_channel': (seam * 32768).tolist(),
    'maximum_interior_adjacent_delta': float(np.max(np.abs(np.diff(actual, axis=0)))),
    'clipped_sample_count': int(np.count_nonzero(np.abs(actual) >= 1.0)),
    'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'limitations': [
        'No subjective listening or professional music-quality acceptance established.',
        'No game mix, engine integration, long-duration repetition/fatigue or buyer fit assessed.',
        'Seam measurements describe sample-level boundary differences; they do not establish perceptual seamlessness.',
        'No buyer audio, names, game assets, commercial song, external samples or vocals used.',
        'Only a short generic ambient probe; no complete save-room or chase composition delivered.',
    ],
}
(ROOT / 'actual-structural-qa.json').write_text(json.dumps(qa, indent=2) + '\n', encoding='utf-8')
print(json.dumps(qa, indent=2))
