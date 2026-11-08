"""Corações que se multiplicam — modelo da caixa de réguas.
Duas barras com abas articuladas (engrenagens nas dobradiças) descem e formam uma caixa;
a caixa estreita e esmaga os corações presos, que estouram em mais corações de outra cor.
Lotou -> borda balança -> mais 3 rodadas estufando e tremendo -> explode."""
import sys, os, math, json, subprocess
import numpy as np
import cv2
from scipy.spatial import cKDTree

W, H, FPS = 720, 1280, 30
DUR = 60.0
NF = int(DUR * FPS)
C = np.array([360.0, 640.0])
R0 = 292.0
RB = 9.5
TB = 18.0          # espessura da barra
TF = 14.0          # espessura da aba
BOX_H = 150.0
BOX_W = 380.0
W_MIN = 4.0
T_CYC = 3.6
TC0 = 1.75
SUB = 3
DT = 1.0 / (FPS * SUB)
MULT = int(os.environ.get("MULT", "6"))
CAP = int(os.environ.get("CAP", "900"))
R_LEVELS = [R0 + 5, R0 + 20, R0 + 36, R0 + 52]

COLORS = [  # BGR — paleta de corações
    (60, 40, 235),    # vermelho
    (180, 110, 255),  # rosa
    (40, 210, 255),   # amarelo
    (210, 70, 160),   # roxo
    (30, 150, 255),   # laranja
    (90, 210, 70),    # verde
    (150, 30, 255),   # pink choque
    (235, 150, 40),   # azul
    (190, 200, 30),   # turquesa
    (225, 160, 215),  # lilás
    (80, 120, 255),   # coral
    (60, 235, 180),   # lima
    (220, 50, 120),   # violeta
    (30, 190, 245),   # dourado
    (240, 200, 90),   # azul-claro
]

# ---------------- mecanismo ----------------
crng = np.random.default_rng(11)
CYCLES = []
k = 0
while TC0 + k * T_CYC < DUR + 4:
    tc = TC0 + k * T_CYC
    bx = C[0] + crng.uniform(-55, 55)
    by = C[1] + crng.uniform(-75, 75)
    if k == 0:
        bx, by = C[0] + 20, C[1] - 20
    CYCLES.append(dict(tc=tc, bx=bx, by=by,
                       far_t=(C[0] + crng.uniform(170, 240), by - BOX_H / 2 - crng.uniform(70, 150)),
                       far_b=(C[0] - crng.uniform(170, 240), by + BOX_H / 2 + crng.uniform(70, 150)),
                       out_t=(C[0] + crng.uniform(180, 260), C[1] + crng.uniform(-200, 40)),
                       out_b=(C[0] - crng.uniform(180, 260), C[1] + crng.uniform(-40, 200))))
    k += 1


def sstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def build_keys():
    top, bot = [], []   # (t, x, y, fold, ease)
    for cy in CYCLES:
        tc, bx, by = cy["tc"], cy["bx"], cy["by"]
        yt, yb = by - BOX_H / 2, by + BOX_H / 2
        top += [(tc - 1.3, *cy["far_t"], 0, "s"), (tc - 0.75, bx - BOX_W / 2, yt - 95, 0, "s"),
                (tc - 0.45, bx - BOX_W / 2, yt, 1, "s"), (tc, bx - W_MIN / 2 - TF / 2, yt, 1, "in"),
                (tc + 0.45, bx - BOX_W * 0.42, yt, 1, "out"), (tc + 0.8, bx - BOX_W * 0.42, yt - 95, 0, "s"),
                (tc + 1.45, *cy["out_t"], 0, "s")]
        bot += [(tc - 1.3, *cy["far_b"], 0, "s"), (tc - 0.75, bx + BOX_W / 2, yb + 95, 0, "s"),
                (tc - 0.45, bx + BOX_W / 2, yb, 1, "s"), (tc, bx + W_MIN / 2 + TF / 2, yb, 1, "in"),
                (tc + 0.45, bx + BOX_W * 0.42, yb, 1, "out"), (tc + 0.8, bx + BOX_W * 0.42, yb + 95, 0, "s"),
                (tc + 1.45, *cy["out_b"], 0, "s")]
    return top, bot


KEYS_T, KEYS_B = build_keys()
KT = np.array([k[0] for k in KEYS_T])


def interp(keys, t):
    i = int(np.searchsorted(KT, t, side="right")) - 1
    if i < 0:
        k = keys[0]; return k[1], k[2], k[3]
    if i >= len(keys) - 1:
        k = keys[-1]; return k[1], k[2], k[3]
    a, b = keys[i], keys[i + 1]
    x = (t - a[0]) / (b[0] - a[0])
    e = b[4]
    if e == "in":
        x = x ** 2.2
    elif e == "out":
        x = 1 - (1 - x) ** 2
    else:
        x = sstep(x)
    return a[1] + (b[1] - a[1]) * x, a[2] + (b[2] - a[2]) * x, a[3] + (b[3] - a[3]) * x


FLAP_L = BOX_H - 6


def mech_at(t):
    xt, yt, ft = interp(KEYS_T, t)
    xb, yb, fb = interp(KEYS_B, t)
    Pt, Pb = np.array([xt, yt]), np.array([xb, yb])
    at = math.pi - ft * math.pi / 2       # aba de cima: esquerda -> para baixo
    ab = -fb * math.pi / 2                # aba de baixo: direita -> para cima
    Et = Pt + FLAP_L * np.array([math.cos(at), math.sin(at)])
    Eb = Pb + FLAP_L * np.array([math.cos(ab), math.sin(ab)])
    segs = [(Pt, Pt + np.array([1000.0, 0]), TB), (Pt, Et, TF),
            (Pb + np.array([-1000.0, 0]), Pb, TB), (Pb, Eb, TF)]
    return Pt, Pb, Et, Eb, ft, fb, segs


def push_capsule(pos, mask, a, b, th):
    ab = b - a
    L2 = ab @ ab
    ap = pos - a
    s = np.clip((ap @ ab) / L2, 0, 1)
    cp = a + s[:, None] * ab
    d = pos - cp
    dist = np.linalg.norm(d, axis=1)
    lim = RB + th / 2
    hit = mask & (dist < lim)
    if hit.any():
        n = d[hit] / np.maximum(dist[hit], 1e-4)[:, None]
        zero = dist[hit] < 1e-3
        if zero.any():
            nn = np.array([-ab[1], ab[0]]) / math.sqrt(L2)
            n[zero] = nn
        pos[hit] = cp[hit] + n * lim


# ---------------- simulação ----------------
def simulate(seed, keep=True):
    rng = np.random.default_rng(seed)
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2))
    col = np.zeros(0, np.int32); rot = np.zeros(0, np.int32)
    trapped = np.zeros(0, bool); ghost = np.zeros(0, bool)
    frames, audio, glows = [], [], []
    st = dict(R=R0, exploded=False, t_over=None, t_expl=None, t_restart=None, ci=0, lvl=0, t_lastpop=None,
              restart0=None)
    frags, spawn_q = [], []
    closed_done, pop_done = set(), set()
    sp_par = [0]

    def add(n, center, dirv, ci):
        nonlocal pos, prev, col, rot, trapped, ghost
        base = math.atan2(dirv[1], dirv[0])
        ang = base + rng.normal(0, 0.5, n)
        rad = rng.uniform(0, 1, n) ** 0.5 * min(45, 6 + n * 0.1)
        p = center + np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
        sp = rng.uniform(160, 320, n)
        v = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        col = np.concatenate([col, np.full(n, ci, np.int32)])
        rot = np.concatenate([rot, rng.integers(0, 5, n).astype(np.int32)])
        trapped = np.concatenate([trapped, np.zeros(n, bool)])
        ghost = np.concatenate([ghost, np.zeros(n, bool)])

    def place_single(cy, ci):
        nonlocal pos, prev, col, rot, trapped, ghost
        pos = np.array([[cy["bx"] - 40, cy["by"] + 5]]); prev = pos.copy()
        col = np.array([ci], np.int32); rot = np.array([2], np.int32)
        trapped = np.zeros(1, bool); ghost = np.ones(1, bool)

    place_single(CYCLES[0], 0)
    t_tick, tick_i = 0.5, 0

    for fi in range(NF):
        t = fi / FPS
        ex = st["exploded"]
        # relógio
        while t_tick <= t:
            if not (st["t_expl"] and st["t_expl"] <= t_tick < (st["t_restart"] or 99)):
                audio.append((t_tick, "tick" if tick_i % 2 == 0 else "tock", 0.22)); tick_i += 1
            if st["t_over"] is not None and not ex:
                t_tick += 0.07 if st["lvl"] >= 3 else 0.5 / (1 + st["lvl"] * 1.3)
            else:
                t_tick += 0.5
        Pt, Pb, Et, Eb, ft, fb, segs = mech_at(t)

        if not ex:
            for k, cy in enumerate(CYCLES):
                tclose = cy["tc"] - 0.45
                if k not in closed_done and t >= tclose:
                    closed_done.add(k)
                    if len(pos):
                        x0, x1 = Pt[0] + TF / 2, Pb[0] - TF / 2
                        y0, y1 = Pt[1] + TB / 2, Pb[1] - TB / 2
                        ins = (pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 1] > y0) & (pos[:, 1] < y1)
                        trapped |= ins
                        ghost[:] = False
                        audio.append((tclose, "clack", 0.5))
                # esmagou: a caixa ficou estreita demais
                if k in closed_done and k not in pop_done and t >= cy["tc"] - 0.04:
                    pop_done.add(k)
                    nt = int(trapped.sum())
                    if nt:
                        km = ~trapped
                        pos, prev, col, rot, ghost = pos[km], prev[km], col[km], rot[km], ghost[km]
                        trapped = trapped[km]
                        st["ci"] = (st["ci"] + 1) % len(COLORS); ci = st["ci"]
                        n = nt * MULT
                        if st["t_over"] is not None:
                            st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                            capn = int(0.86 * (R_LEVELS[st["lvl"]] / RB) ** 2)
                            n = max(nt, min(n, capn - len(pos)))
                            audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                        else:
                            n = max(nt, min(n, CAP + 30 - len(pos)))
                        cen = (Pt + Pb) / 2
                        pulses = max(1, min(14, (n + 3) // 4))
                        base, rest = divmod(n, pulses)
                        for p in range(pulses):
                            cnt = base + (1 if p < rest else 0)
                            up = (p % 2 == 0)
                            c0 = np.array([cen[0], Pt[1] - TB / 2 - RB - 4]) if up else np.array([cen[0], Pb[1] + TB / 2 + RB + 4])
                            dv = np.array([-0.6, -1.0]) if up else np.array([0.6, 1.0])
                            spawn_q.append([t + p * 0.075, cnt, c0, dv, ci])
                        glows.append((t, cen.copy(), ci))
                        audio.append((t, "burst", 0.8))
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, c0, dv, ci = spawn_q.pop(0)
            if not st["exploded"] and n > 0:
                add(n, c0, dv, ci)
                audio.append((ts, "stick" if sp_par[0] % 2 == 0 else "stock", 0.6)); sp_par[0] += 1

        # lotação e explosão
        if not ex and st["t_over"] is None and len(pos) >= CAP:
            st["t_over"] = t; audio.append((t, "rumble", 0.5))
        if not ex and st["t_over"] is not None:
            st["R"] += (R_LEVELS[st["lvl"]] - st["R"]) * 0.06
            if st["lvl"] >= 3 and t >= st["t_lastpop"] + 1.0:
                st.update(exploded=True, t_expl=t, t_restart=t + 2.3); ex = True
                audio.append((t, "boom", 1.0))
                d = pos - C
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                spd = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / st["R"])
                prev = pos - (d / dist * spd + rng.normal(0, 120, pos.shape)) * DT
                for q in range(22):
                    a0 = q * 2 * math.pi / 22; a1 = a0 + 2 * math.pi / 22 * 0.85; am = (a0 + a1) / 2
                    frags.append(dict(c=C.copy(), r=st["R"], a0=a0, a1=a1, rot=0.0,
                                      v=np.array([math.cos(am), math.sin(am)]) * rng.uniform(500, 1100),
                                      w=rng.uniform(-6, 6)))
        if ex and t >= st["t_restart"]:
            st.update(exploded=False, t_over=None, R=R0, lvl=0, t_lastpop=None, restart0=t); ex = False
            frags = []
            nxt = [cy for cy in CYCLES if cy["tc"] - 0.45 > t + 0.8]
            st["ci"] = (st["ci"] + 1) % len(COLORS)
            place_single(nxt[0], st["ci"])
            audio.append((t, "pop", 0.6))
            for k, cy in enumerate(CYCLES):
                if cy["tc"] - 0.45 <= t:
                    closed_done.add(k); pop_done.add(k)

        # física
        grav = np.array([0.0, 900.0]) if ex else np.zeros(2)
        for s in range(SUB):
            if len(pos) == 0:
                break
            ts = t + s * DT
            v = (pos - prev) / DT * 0.9993
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            cap = 2500 if ex else 700
            v = np.where(spd > cap, v / np.maximum(spd, 1e-6) * cap, v)
            prev = pos; pos = pos + v * DT + grav * DT * DT
            Pt, Pb, Et, Eb, ft, fb, segs = mech_at(ts)
            for it in range(2):
                pr = cKDTree(pos).query_pairs(2 * RB, output_type="ndarray")
                if len(pr):
                    i, j = pr[:, 0], pr[:, 1]
                    d = pos[j] - pos[i]
                    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4)
                    corr = d * ((2 * RB - dist) / dist * 0.5 * 0.85)[:, None]
                    dp = np.zeros_like(pos)
                    np.add.at(dp, i, -corr); np.add.at(dp, j, corr)
                    pos += dp
                if not ex:
                    solid = ~ghost & ~trapped
                    for a, b, th in segs:
                        push_capsule(pos, solid, a, b, th)
                    if trapped.any():
                        x0, x1 = Pt[0] + TF / 2 + RB, Pb[0] - TF / 2 - RB
                        y0, y1 = Pt[1] + TB / 2 + RB, Pb[1] - TB / 2 - RB
                        if x1 < x0:
                            x0 = x1 = (x0 + x1) / 2
                        pos[trapped, 0] = np.clip(pos[trapped, 0], x0, x1)
                        pos[trapped, 1] = np.clip(pos[trapped, 1], y0, max(y0, y1))
                    d = pos - C
                    dist = np.linalg.norm(d, axis=1)
                    out = dist > st["R"] - RB
                    if out.any():
                        nrm = d[out] / dist[out][:, None]
                        pos[out] = C + nrm * (st["R"] - RB)
                        vv = pos[out] - prev[out]
                        vn = (vv * nrm).sum(1, keepdims=True)
                        prev[out] = pos[out] - (vv - np.maximum(vn, 0) * nrm * 1.85)
        if ex:
            kp = (pos[:, 0] > -50) & (pos[:, 0] < W + 50) & (pos[:, 1] < H + 50) & (pos[:, 1] > -500)
            pos, prev, col, rot, trapped, ghost = pos[kp], prev[kp], col[kp], rot[kp], trapped[kp], ghost[kp]
            for fr in frags:
                fr["c"] = fr["c"] + fr["v"] / FPS
                fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS
                fr["rot"] += fr["w"] / FPS
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), col=col.copy(), rot=rot.copy(), R=st["R"], exploded=ex,
                               t_over=st["t_over"], lvl=st["lvl"], t_lastpop=st["t_lastpop"],
                               t_expl=st["t_expl"], restart0=st["restart0"], frags=[dict(f) for f in frags]))
    return frames, audio, glows, st


# ---------------- desenho ----------------
def heart_sprite(color, angle_deg, size=26, ss=6):
    S = size * ss
    img = np.zeros((S, S, 4), np.float32)
    tt = np.linspace(0, 2 * math.pi, 400)
    hx = 16 * np.sin(tt) ** 3
    hy = -(13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt))
    sc = (size * 0.46) / 17.0 * ss
    a = math.radians(angle_deg)
    px = hx * math.cos(a) - hy * math.sin(a)
    py = hx * math.sin(a) + hy * math.cos(a)
    pts = np.stack([S / 2 + px * sc, S / 2 + py * sc + 1.0 * ss], 1).astype(np.int32)
    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    m = mask.astype(np.float32) / 255
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    nx, ny = (xx - S / 2) / (S / 2), (yy - S / 2) / (S / 2)
    col = np.array(color, np.float32)
    shade = np.clip(1.15 - 0.45 * (ny + 0.3) - 0.15 * nx, 0.55, 1.25)
    rgb = col[None, None] * shade[..., None]
    # contorno escuro
    er = cv2.erode(mask, np.ones((ss * 2, ss * 2), np.uint8))
    edge = (m - er.astype(np.float32) / 255)[..., None]
    rgb = rgb * (1 - edge * 0.45)
    # brilho
    hlc = (S / 2 + (-5.5 * math.cos(a) - (-5) * math.sin(a)) * sc, S / 2 + (-5.5 * math.sin(a) + (-5) * math.cos(a)) * sc)
    hl = np.zeros((S, S), np.uint8)
    cv2.ellipse(hl, (int(hlc[0]), int(hlc[1])), (int(3.2 * sc), int(2.0 * sc)), angle_deg - 35, 0, 360, 255, -1)
    hl = cv2.GaussianBlur(hl.astype(np.float32) / 255, (0, 0), ss * 0.8) * m
    rgb = rgb + (255 - rgb) * hl[..., None] * 0.85
    rgba = np.dstack([np.clip(rgb, 0, 255), m * 255])
    small = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
    al = small[..., 3:4] / 255
    return (small[..., :3] * al).astype(np.float32), (1 - al).astype(np.float32)


ROTS = [-22, -11, 0, 11, 22]
SPR = [[heart_sprite(c, r) for r in ROTS] for c in COLORS]
SS = SPR[0][0][0].shape[0]
HALF = SS // 2
SH = 16


def P(p):
    return (int(round(p[0] * SH)), int(round(p[1] * SH)))


def make_glow(r):
    s = int(r * 2)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    return np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 2


GLOW = make_glow(170)


def add_glow(img, center, color, k):
    s = GLOW.shape[0]
    x0, y0 = int(center[0] - s / 2), int(center[1] - s / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if xa >= xb or ya >= yb or k <= 0.01:
        return
    gg = GLOW[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * k
    reg = img[ya:yb, xa:xb].astype(np.float32) + gg * (np.array(color, np.float32) * 0.8 + 60)
    img[ya:yb, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)


def gear_poly(center, ang, r, teeth=10):
    pts = []
    for i in range(teeth * 4):
        a = ang + i * (2 * math.pi / (teeth * 4))
        rr = r if (i % 4) in (0, 1) else r * 0.76
        pts.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a)))
    return (np.array(pts) * SH).astype(np.int32)


def draw_bar(img, a, b, th, flap=False):
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1:
        return
    u = d / L; n = np.array([-u[1], u[0]])
    h = th / 2
    poly = np.array([a + n * h, b + n * h, b - n * h, a - n * h])
    cv2.fillPoly(img, [(poly * SH).astype(np.int32)], (150, 148, 145), cv2.LINE_AA, 4)
    cv2.line(img, P(a + n * (h - 2.5)), P(b + n * (h - 2.5)), (110, 108, 105), 2, cv2.LINE_AA, 4)
    cv2.line(img, P(a - n * (h - 2.5)), P(b - n * (h - 2.5)), (205, 203, 200), 2, cv2.LINE_AA, 4)
    cv2.polylines(img, [(poly * SH).astype(np.int32)], True, (80, 80, 80), 1, cv2.LINE_AA, 4)
    if flap:
        cv2.line(img, P(a + u * 12 + n * 2), P(b - u * 6 + n * 2), (60, 170, 230), 2, cv2.LINE_AA, 4)
    else:
        for s in np.arange(46, min(L, 900), 46):
            cv2.circle(img, P(a + u * s), int(1.8 * SH), (95, 95, 95), -1, cv2.LINE_AA, 4)


def render(fr, glows, srng):
    t = fr["t"]; ex = fr["exploded"]
    img = np.zeros((H, W, 3), np.uint8)
    act = []
    for (t0, gc, ci) in glows:
        if t0 - 0.02 <= t < t0 + 1.0:
            k = 1.0 if t < t0 + 0.15 else max(0.0, 1 - (t - t0 - 0.15) / 0.85)
            act.append((gc, ci, k, t - t0)); add_glow(img, gc, COLORS[ci], k * 0.9)
    pos, col, rot = fr["pos"], fr["col"], fr["rot"]
    f = img.astype(np.float32)
    ip = np.round(pos).astype(np.int32)
    for k in range(len(pos)):
        x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
        pre, inv = SPR[col[k]][rot[k]]
        if x0 < 0 or y0 < 0 or x0 + SS > W or y0 + SS > H:
            if x0 + SS <= 0 or y0 + SS <= 0 or x0 >= W or y0 >= H:
                continue
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + SS), min(H, y0 + SS)
            sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * inv[sl] + pre[sl]
            continue
        reg = f[y0:y0 + SS, x0:x0 + SS]; reg *= inv; reg += pre
    img = np.clip(f, 0, 255).astype(np.uint8)

    r0 = fr["restart0"]
    fade = 1.0 if r0 is None else min(1.0, max(0.0, (t - r0) / 0.5))
    wob = 0.0
    if fr["t_over"] is not None and not ex:
        lv = fr["lvl"]
        kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.6) if fr["t_lastpop"] is not None else 0.0
        kick0 = max(0.0, 1 - (t - fr["t_over"]) / 0.8)
        wob = [3, 6, 10, 15][lv] + 10 * kick + 8 * kick0
        if lv >= 3:
            wob += 10 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
    if not ex:
        ov = img.copy()
        if wob > 0:
            angs = np.linspace(0, 2 * math.pi, 240, endpoint=False)
            rr = fr["R"] + wob * (np.sin(3 * angs + t * 23) * 0.6 + np.sin(5 * angs - t * 31) * 0.4)
            pts = np.stack([C[0] + rr * np.cos(angs), C[1] + rr * np.sin(angs)], 1)
            cv2.polylines(ov, [(pts * SH).astype(np.int32)], True, (255, 255, 255), 3, cv2.LINE_AA, 4)
        else:
            cv2.circle(ov, P(C), int(fr["R"] * SH), (255, 255, 255), 3, cv2.LINE_AA, 4)
        img = ov if fade >= 1 else cv2.addWeighted(ov, fade, img, 1 - fade, 0)
    for fg in fr["frags"]:
        pts = np.array([[fg["c"][0] + fg["r"] * math.cos(a), fg["c"][1] + fg["r"] * math.sin(a)]
                        for a in np.linspace(fg["a0"], fg["a1"], 8)])
        cen = pts.mean(0); cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"])
        pts = (pts - cen) @ np.array([[cr, sr], [-sr, cr]]) + cen
        cv2.polylines(img, [(pts * SH).astype(np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA, 4)

    # barras, abas e engrenagens
    ta = 1.0
    if ex:
        ta = max(0.0, 1 - (t - fr["t_expl"]) / 0.25)
    elif r0 is not None:
        ta = fade
    if ta > 0:
        Pt, Pb, Et, Eb, ft, fb, segs = mech_at(t)
        ov = img.copy()
        draw_bar(ov, Pt, Pt + np.array([1000.0, 0]), TB)
        draw_bar(ov, Pb + np.array([-1000.0, 0]), Pb, TB)
        draw_bar(ov, Pt, Et, TF, flap=True)
        draw_bar(ov, Pb, Eb, TF, flap=True)
        cv2.putText(ov, "NewsLab", (int(Pt[0] + 22), int(Pt[1] + 6)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        gang = int(t / 0.5) * math.radians(36) + min(1, (t % 0.5) * 10) * math.radians(36)
        for Pp, sgn in ((Pt, 1), (Pb, -1)):
            gp = gear_poly(Pp, sgn * gang, 15)
            cv2.fillPoly(ov, [gp], (215, 215, 215), cv2.LINE_AA, 4)
            cv2.polylines(ov, [gp], True, (80, 80, 80), 1, cv2.LINE_AA, 4)
            cv2.circle(ov, P(Pp), int(6 * SH), (120, 120, 120), -1, cv2.LINE_AA, 4)
            cv2.circle(ov, P(Pp), int(2.5 * SH), (50, 50, 50), -1, cv2.LINE_AA, 4)
        img = ov if ta >= 1 else cv2.addWeighted(ov, ta, img, 1 - ta, 0)
    for (gc, ci, k, dtt) in act:
        if k > 0.05:
            ov = img.copy()
            cv2.circle(ov, P(gc), int((14 + dtt * 170) * SH), tuple(int(min(255, x * 0.6 + 110)) for x in COLORS[ci]), 3, cv2.LINE_AA, 4)
            img = cv2.addWeighted(ov, k * 0.8, img, 1 - k * 0.8, 0)
    if ex:
        fl = max(0.0, 0.85 * (1 - (t - fr["t_expl"]) / 0.35))
        if fl > 0:
            img = cv2.addWeighted(img, 1 - fl, np.full_like(img, 255), fl, 0)
        shake = max(0.0, 18 * (1 - (t - fr["t_expl"]) / 0.7))
    elif fr["t_over"] is not None:
        lv = fr["lvl"]
        kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.5) if fr["t_lastpop"] is not None else 0.0
        kick0 = max(0.0, 1 - (t - fr["t_over"]) / 0.6)
        shake = [1.0, 2.5, 4.0, 6.0][lv] + 7 * kick + 4 * kick0
        if lv >= 3:
            shake += 6 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
    else:
        shake = 0
    if shake > 0.5:
        M = np.float32([[1, 0, srng.normal() * shake], [0, 1, srng.normal() * shake]])
        img = cv2.warpAffine(img, M, (W, H))
    return img


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "search":
        for seed in range(int(sys.argv[2]), int(sys.argv[3])):
            _, au, gl, st = simulate(seed, keep=False)
            full = [a[0] for a in au if a[1] == "rumble"]
            boom = [a[0] for a in au if a[1] == "boom"]
            print(seed, "cheio", round(full[0], 1) if full else None, "boom", [round(b, 1) for b in boom],
                  "estouros", [round(g[0], 1) for g in gl][:8], flush=True)
        sys.exit()
    seed = int(sys.argv[2])
    frames, audio, glows, st = simulate(seed)
    r0 = st["restart0"]
    if r0 is not None:
        later = [g[0] for g in glows if g[0] > r0]
        t_end = (later[0] + 1.4) if later else r0 + 4.0
    else:
        t_end = DUR
    frames = [f for f in frames if f["t"] < t_end]
    audio = [a for a in audio if a[0] < t_end]
    print("duracao", len(frames) / FPS, "explosao", st["t_expl"], flush=True)
    srng = np.random.default_rng(1)
    if mode == "preview":
        for fi in [int(x) for x in sys.argv[3:]]:
            if fi < len(frames):
                cv2.imwrite(f"out/h_{fi:04d}.png", render(frames[fi], glows, srng))
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "out/h_video_tmp.mp4"], stdin=subprocess.PIPE)
        for fr in frames:
            enc.stdin.write(render(fr, glows, srng).tobytes())
        enc.stdin.close(); enc.wait()
        json.dump(dict(dur=len(frames) / FPS, events=sorted(audio)), open("out/h_audio_events.json", "w"))
        print("ok", enc.returncode)
