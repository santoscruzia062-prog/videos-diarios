import json, wave
import numpy as np
from scipy.signal import lfilter

SR = 44100
data = json.load(open("out/d_audio_events.json"))
DUR = data["dur"]
N = int(SR * DUR)
out = np.zeros((N, 2), np.float32)
rng = np.random.default_rng(5)


def env(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def lowpass(x, a):
    return lfilter([a], [1, a - 1], x)


def click(f1, f2, tau, ms=90):
    n = int(SR * ms / 1000); t = np.arange(n) / SR
    s = (0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)) * env(n, tau)
    return s + rng.normal(0, 1, n) * env(n, 0.0015) * 0.5


def coin_hit(f0, tau):
    """batida de moeda: parciais inarmônicas de disco metálico"""
    n = int(SR * 0.45); t = np.arange(n) / SR
    s = np.zeros(n)
    for ratio, amp, dec in ((1.0, 1.0, 1.0), (2.32, 0.7, 0.8), (4.25, 0.45, 0.55), (6.63, 0.3, 0.4), (9.1, 0.18, 0.3)):
        s += amp * np.sin(2 * np.pi * f0 * ratio * t + rng.uniform(0, 6)) * env(n, tau * dec)
    s += rng.normal(0, 1, n) * env(n, 0.0012) * 0.6
    return s


def coin():
    """moedas contando: 2-3 moedas batendo uma na outra"""
    n = int(SR * 0.6)
    s = np.zeros(n)
    k = rng.integers(2, 4)
    t0 = 0
    for i in range(k):
        h = coin_hit(rng.uniform(2300, 3600), rng.uniform(0.05, 0.12)) * (1 - i * 0.2)
        d = int(t0 * SR)
        s[d:d + len(h)] += h[: n - d]
        t0 += rng.uniform(0.025, 0.06)
    return s


def coinburst():
    n = int(SR * 0.9)
    s = np.zeros(n)
    for i in range(14):
        h = coin_hit(rng.uniform(2200, 4200), rng.uniform(0.04, 0.12)) * rng.uniform(0.4, 1.0)
        d = int(rng.uniform(0, 0.45) ** 1.4 * SR)
        s[d:d + len(h)] += h[: n - d]
    return s


def press():
    n = int(SR * 0.3); t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (150 - 70 * t / 0.3) * t) * env(n, 0.06)
    metal = (np.sin(2 * np.pi * 820 * t) + 0.5 * np.sin(2 * np.pi * 1370 * t)) * env(n, 0.025) * 0.5
    return thump + metal + rng.normal(0, 1, n) * env(n, 0.005) * 0.5


def clack():
    n = int(SR * 0.15); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1250 * t) + 0.6 * np.sin(2 * np.pi * 2300 * t)) * env(n, 0.018) + rng.normal(0, 1, n) * env(n, 0.003) * 0.6


def kaching():
    """caixa registradora: 'ka' (gaveta) + 'ching' (sino) duas vezes + chuva de moedas"""
    n = int(SR * 3.0); t = np.arange(n) / SR
    s = np.zeros(n)
    ka = lowpass(rng.normal(0, 1, n), 0.25) * env(n, 0.03) * 2.2 + np.sin(2 * np.pi * 180 * t) * env(n, 0.05)
    s += ka
    for start, f0, amp in ((0.09, 2093.0, 1.0), (0.30, 2637.0, 0.8)):
        d = int(start * SR); tt = t[: n - d]
        bell = np.zeros(len(tt))
        for ratio, a, dec in ((1.0, 1.0, 1.0), (1.003, 0.8, 1.0), (2.0, 0.5, 0.6), (2.76, 0.35, 0.45), (4.07, 0.25, 0.3), (5.4, 0.15, 0.2)):
            bell += a * np.sin(2 * np.pi * f0 * ratio * tt) * np.exp(-tt / (0.9 * dec))
        s[d:] += bell * amp * 0.8
    for i in range(40):
        h = coin_hit(rng.uniform(2000, 4500), rng.uniform(0.05, 0.14)) * rng.uniform(0.25, 0.7)
        d = int((0.35 + rng.uniform(0, 1.6) ** 1.3) * SR)
        if d < n:
            s[d:d + len(h)] += h[: n - d]
    return s


def boom():
    n = int(SR * 2.4); t = np.arange(n) / SR
    nz = rng.normal(0, 1, n)
    return lowpass(nz, 0.02) * 6 * env(n, 0.5) + lowpass(nz, 0.35) * env(n, 0.08) + \
        np.sin(2 * np.pi * np.cumsum(70 - 40 * np.minimum(t, 1)) / SR) * env(n, 0.55) * 0.9


def rumble():
    n = int(SR * 2.0); t = np.arange(n) / SR
    e = (t / 2.0) ** 2
    return lowpass(rng.normal(0, 1, n), 0.01) * 8 * e + np.sin(2 * np.pi * 55 * t) * e * 0.4


BASE = {"tick": click(3100, 4700, 0.006), "tock": click(1750, 2650, 0.009), "press": press(), "clack": clack(),
        "coinburst": coinburst(), "kaching": kaching(), "boom": boom(), "rumble": rumble()}
COIN_VARIANTS = [coin() for _ in range(10)]
for k in list(BASE):
    BASE[k] = BASE[k] / (np.abs(BASE[k]).max() + 1e-9)
COIN_VARIANTS = [c / np.abs(c).max() for c in COIN_VARIANTS]

for i, (t, kind, vol) in enumerate(data["events"]):
    s = (COIN_VARIANTS[i % 10] if kind == "coin" else BASE[kind]) * vol * {"coin": 0.45, "coinburst": 0.5, "kaching": 2.2, "boom": 1.3}.get(kind, 1.0)
    i0 = int(t * SR)
    if i0 >= N:
        continue
    seg = s[: N - i0]
    pan = {"tick": 0.2, "tock": -0.2, "coin": rng.uniform(-0.5, 0.5)}.get(kind, 0.0)
    out[i0:i0 + len(seg), 0] += seg * (1 - pan)
    out[i0:i0 + len(seg), 1] += seg * (1 + pan)

for d, g in ((0.045, 0.22), (0.11, 0.13), (0.19, 0.07)):
    k = int(d * SR)
    out[k:] += out[:-k].copy() * g
out /= np.abs(out).max() / 0.9
w = wave.open("out/d_audio.wav", "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((out * 32767).astype(np.int16).tobytes()); w.close()
print("ok", len(data["events"]))
