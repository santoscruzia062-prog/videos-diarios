import json, wave
import numpy as np

SR = 44100
DUR = 51.4
N = int(SR * DUR)
out = np.zeros((N, 2), np.float32)
rng = np.random.default_rng(3)


def env(n, tau):
    t = np.arange(n) / SR
    return np.exp(-t / tau)


def click(f1, f2, tau, n_ms=90):
    n = int(SR * n_ms / 1000)
    t = np.arange(n) / SR
    s = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)
    s *= env(n, tau)
    nz = rng.normal(0, 1, n) * env(n, 0.0015) * 0.6
    s = s + nz
    a = np.minimum(1, np.arange(n) / (SR * 0.0005))
    return s * a


TICK = click(3100, 4700, 0.008)
TOCK = click(1750, 2650, 0.012)
STICK = click(2600, 5200, 0.012, 120) * 1.0
STOCK = click(1400, 2100, 0.016, 120) * 1.0


def press():
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (140 - 60 * t / 0.35) * t) * env(n, 0.07)
    metal = (np.sin(2 * np.pi * 920 * t) + 0.5 * np.sin(2 * np.pi * 1530 * t)) * env(n, 0.03) * 0.5
    nz = rng.normal(0, 1, n) * env(n, 0.006) * 0.5
    return thump + metal + nz


def pop():
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    f = 380 + 900 * t / 0.12
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.035)


def lowpass(x, a):
    y = np.zeros_like(x)
    acc = 0.0
    # filtro de 1 polo
    from scipy.signal import lfilter
    return lfilter([a], [1, a - 1], x)


def boom():
    n = int(SR * 2.6)
    t = np.arange(n) / SR
    nz = rng.normal(0, 1, n)
    rumble = lowpass(nz, 0.02) * 6 * env(n, 0.55)
    crack = lowpass(nz, 0.35) * env(n, 0.09) * 1.2
    sub = np.sin(2 * np.pi * np.cumsum(70 - 40 * np.minimum(t, 1)) / SR) * env(n, 0.6) * 0.9
    return rumble + crack + sub


def rumble():
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    nz = lowpass(rng.normal(0, 1, n), 0.01) * 8
    e = (t / 2.0) ** 2
    return nz * e + np.sin(2 * np.pi * 55 * t) * e * 0.4


def clink():
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2350 * t) + 0.6 * np.sin(2 * np.pi * 3720 * t) + 0.3 * np.sin(2 * np.pi * 5100 * t)) * env(n, 0.05)


SOUNDS = {"clink": clink(), "tick": TICK, "tock": TOCK, "stick": STICK, "stock": STOCK, "press": press(), "pop": pop(),
          "boom": boom(), "rumble": rumble()}
for k in SOUNDS:
    s = SOUNDS[k]
    SOUNDS[k] = s / (np.abs(s).max() + 1e-9)

ev = json.load(open("out/audio_events.json"))
for i, (t, kind, vol) in enumerate(ev):
    s = SOUNDS[kind] * vol
    if kind in ("stick", "stock"):
        s = s + SOUNDS["pop"] * 0.18 * vol
    i0 = int(t * SR)
    if i0 >= N:
        continue
    seg = s[: N - i0]
    pan = 0.0 if kind in ("boom", "rumble", "press", "clink") else (0.25 if kind in ("tick", "stick") else -0.25)
    out[i0:i0 + len(seg), 0] += seg * (1 - pan)
    out[i0:i0 + len(seg), 1] += seg * (1 + pan)

# um pouco de ambiência (eco curto)
for d, g in ((0.045, 0.25), (0.11, 0.15), (0.19, 0.08)):
    k = int(d * SR)
    out[k:] += out[:-k] * g * np.array([0.8, 1.2]) if False else out[:-k].copy() * g

out /= np.abs(out).max() / 0.9
pcm = (out * 32767).astype(np.int16)
w = wave.open("out/audio.wav", "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes(pcm.tobytes())
w.close()
print("ok", len(ev))
