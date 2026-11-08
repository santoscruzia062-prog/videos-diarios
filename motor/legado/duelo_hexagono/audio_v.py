import json, wave
import numpy as np
from scipy.signal import lfilter

SR = 44100
data = json.load(open("out/audio_events.json"))
N = int(SR * data["dur"])
out = np.zeros((N, 2), np.float32)
rng = np.random.default_rng(9)


def env(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def lowpass(x, a):
    return lfilter([a], [1, a - 1], x)


def tone(f, dur, tau, harm=((1, 1.0),), attack=0.002):
    n = int(SR * dur); t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in harm)
    return s * env(n, tau) * np.minimum(1, t / attack)


def click(f1, f2, tau):
    n = int(SR * 0.08); t = np.arange(n) / SR
    return (0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)) * env(n, tau) + rng.normal(0, 1, n) * env(n, 0.0015) * 0.4


def angel(i):
    # "plim" angelical: sino brilhante, nota de uma escala maior
    notes = [1046.5, 1174.7, 1318.5, 1568.0, 1760.0, 2093.0]
    f = notes[i % len(notes)]
    x = tone(f, 0.5, 0.16, ((1, 1), (2, 0.35), (3.01, 0.15)))
    y = tone(f * 2.003, 0.3, 0.06, ((1, 0.3),))
    x[:len(y)] += y
    return x


def devil(i):
    # "dum" diabólico: pluck grave com dente de serra filtrado
    notes = [110.0, 116.5, 130.8, 146.8, 155.6, 98.0]
    f = notes[i % len(notes)]
    n = int(SR * 0.35); t = np.arange(n) / SR
    saw = 2 * ((f * t) % 1) - 1
    s = lowpass(saw, 0.12) * env(n, 0.09) * 2.2 + np.sin(2 * np.pi * f * t) * env(n, 0.12)
    s += np.sin(2 * np.pi * f * 1.5 * t) * env(n, 0.05) * 0.4
    return s


def steal():
    n = int(SR * 0.35); t = np.arange(n) / SR
    f = 1800 * np.exp(-t * 9) + 200
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.12) * (1 + 0.5 * np.sign(np.sin(2 * np.pi * 30 * t)))


def lead():
    s = np.zeros(int(SR * 0.6))
    for d, f in ((0, 1568.0), (0.12, 2093.0)):
        x = tone(f, 0.45, 0.15, ((1, 1), (2, 0.3), (3, 0.1)))
        i0 = int(d * SR); s[i0:i0 + len(x)] += x[: len(s) - i0]
    return s


def press():
    n = int(SR * 0.25); t = np.arange(n) / SR
    return np.sin(2 * np.pi * (140 - 60 * t / 0.25) * t) * env(n, 0.05) + \
        (np.sin(2 * np.pi * 760 * t) + 0.5 * np.sin(2 * np.pi * 1290 * t)) * env(n, 0.02) * 0.4


def clack():
    n = int(SR * 0.12); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1200 * t) + 0.6 * np.sin(2 * np.pi * 2200 * t)) * env(n, 0.015)


def boom():
    n = int(SR * 2.4); t = np.arange(n) / SR
    nz = rng.normal(0, 1, n)
    return lowpass(nz, 0.02) * 6 * env(n, 0.5) + lowpass(nz, 0.35) * env(n, 0.08) + \
        np.sin(2 * np.pi * np.cumsum(70 - 40 * np.minimum(t, 1)) / SR) * env(n, 0.55) * 0.9


def rumble():
    n = int(SR * 2.0); t = np.arange(n) / SR
    e = (t / 2.0) ** 2
    return lowpass(rng.normal(0, 1, n), 0.01) * 8 * e + np.sin(2 * np.pi * 55 * t) * e * 0.4


def fanfare():
    s = np.zeros(int(SR * 3.2))
    seq = [(0.0, [523.3]), (0.14, [659.3]), (0.28, [784.0]), (0.42, [1046.5]),
           (0.75, [523.3, 659.3, 784.0, 1046.5])]
    for d, fs in seq:
        dur = 0.3 if len(fs) == 1 else 2.2
        for f in fs:
            n = int(SR * dur); t = np.arange(n) / SR
            sq = np.sign(np.sin(2 * np.pi * f * t)) * 0.25 + np.sin(2 * np.pi * f * t)
            x = lowpass(sq, 0.3) * np.minimum(1, t / 0.01) * (env(n, 0.25) if len(fs) == 1 else env(n, 0.9))
            vib = 1.0
            i0 = int(d * SR); s[i0:i0 + n] += x[: len(s) - i0] * vib
    return s


BASE = {"tick": click(3100, 4700, 0.006), "tock": click(1750, 2650, 0.009), "press": press(), "clack": clack(),
        "steal": steal(), "lead": lead(), "boom": boom(), "rumble": rumble(), "fanfare": fanfare()}
for k in BASE:
    BASE[k] = BASE[k] / np.abs(BASE[k]).max()
AV = [angel(i) / np.abs(angel(i)).max() for i in range(6)]
DV = [devil(i) / np.abs(devil(i)).max() for i in range(6)]
GAIN = {"angel": 0.3, "devil": 0.36, "boom": 2.4, "fanfare": 1.9, "lead": 1.2, "steal": 0.8, "rumble": 1.3}

ia = idv = 0
for (t, kind, vol) in data["events"]:
    if kind == "angel":
        s = AV[ia % 6]; ia += 1
    elif kind == "devil":
        s = DV[idv % 6]; idv += 1
    else:
        s = BASE[kind]
    s = s * vol * GAIN.get(kind, 1.0)
    i0 = int(t * SR)
    if i0 >= N:
        continue
    seg = s[: N - i0]
    pan = {"angel": -0.35, "devil": 0.35, "tick": 0.15, "tock": -0.15}.get(kind, 0.0)
    out[i0:i0 + len(seg), 0] += seg * (1 - pan)
    out[i0:i0 + len(seg), 1] += seg * (1 + pan)

for d, g in ((0.05, 0.22), (0.12, 0.13), (0.21, 0.07)):
    k = int(d * SR)
    out[k:] += out[:-k].copy() * g
out /= np.abs(out).max() / 0.9
w = wave.open("out/audio.wav", "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((out * 32767).astype(np.int16).tobytes()); w.close()
print("ok", len(data["events"]))
