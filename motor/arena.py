"""Motor genérico de DUELO de emojis em qualquer formato (quadrado, retângulo, círculo, estrela,
coração, octógono, losango).
Rotor central gira sempre no mesmo sentido; cada braço leva uma caixa com a boca para a frente:
ela 'abocanha' emojis, fecha as portinhas, o martelo esmaga, e quem estiver encostado no martelo
leva o bolso todo (x MULT) para o seu time. Placar ao vivo, VIROU!, lotou -> 3 rodadas tremendo
-> explode -> tela do vencedor."""
import sys, os, math, json, subprocess, wave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.signal import lfilter

W, H, FPS = 720, 1280, 30
DUR = 56.0
SUB = 3
DT = 1.0 / (FPS * SUB)
T0 = 0.6                 # início do 1º ciclo
CYC = 1.5                # ciclo das caixas
OMEGA = 0.85             # rotação do rotor (rad/s), sentido horário
SH = 16
FONT_B = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
EMOJI_FONT = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)


# ======================= formatos =======================
def poly_regular(n, apothem, cx, cy, rot):
    R = apothem / math.cos(math.pi / n)
    return [(cx + R * math.cos(rot + 2 * math.pi * k / n), cy + R * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def shape_pts(kind):
    if kind == "square":
        return [(40, 340), (680, 340), (680, 980), (40, 980)]
    if kind == "rect":
        return [(60, 268), (660, 268), (660, 1062), (60, 1062)]
    if kind == "circle":
        return [(360 + 335 * math.cos(a), 665 + 335 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 180, endpoint=False)]
    if kind == "octagon":
        return poly_regular(8, 330, 360, 665, math.pi / 8)
    if kind == "diamond":
        return [(360, 272), (700, 665), (360, 1058), (20, 665)]
    if kind == "pentagon":
        return poly_regular(5, 285, 360, 690, -math.pi / 2)
    if kind == "star":
        pts = []
        for k in range(10):
            r = 350 if k % 2 == 0 else 205
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((360 + r * math.cos(a), 690 + r * math.sin(a)))
        return pts
    if kind == "heart":
        t = np.linspace(0, 2 * math.pi, 240, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        sx = 670 / (x.max() - x.min())
        sy = 700 / (y.max() - y.min())
        s = min(sx, sy)
        x = x * s; y = y * s
        x = x - (x.max() + x.min()) / 2 + 360
        y = y - y.min() + 300
        return list(zip(x, y))
    raise ValueError(kind)


CONFIGS = {
    "pentagono": dict(shape="pentagon", arms=5, names=("TUBARÃO", "JACARÉ"), emo=("\U0001F988", "\U0001F40A"),
                      col=((210, 150, 70), (60, 180, 70)), snd=("chomp", "roar"), bg=(32, 18, 8)),
    "octogono": dict(shape="octagon", arms=4, names=("LEÃO", "LOBO"), emo=("\U0001F981", "\U0001F43A"),
                     col=((40, 110, 235), (200, 170, 150)), snd=("roar", "howl"), bg=(30, 14, 12), posts=True),
    "coracao": dict(shape="heart", arms=3, names=("AMOR", "ÓDIO"), emo=("\U0001F60D", "\U0001F47F"),
                    col=((150, 90, 255), (210, 60, 160)), snd=("love", "angry"), bg=(30, 10, 28), pull=160, box=1.3),
    "estrela": dict(shape="star", arms=5, names=("SOL", "LUA"), emo=("☀️", "\U0001F31A"),
                    col=((40, 200, 255), (200, 150, 110)), snd=("sun", "moon"), bg=(34, 14, 10), pull=110),
    "circulo": dict(shape="circle", arms=3, names=("FUTEBOL", "BASQUETE"), emo=("⚽", "\U0001F3C0"),
                    col=((240, 240, 240), (40, 120, 245)), snd=("kick", "bounce"), bg=(14, 26, 12)),
    "quadrado": dict(shape="square", arms=4, names=("CACHORRO", "GATO"), emo=("\U0001F436", "\U0001F431"),
                     col=((120, 170, 230), (40, 200, 250)), snd=("bark", "meow"), bg=(20, 18, 30)),
    "retangulo": dict(shape="rect", arms=3, names=("PIZZA", "HAMBÚRGUER"), emo=("\U0001F355", "\U0001F354"),
                      col=((60, 80, 230), (40, 160, 230)), snd=("pop", "chomp"), bg=(14, 18, 30)),
    "losango": dict(shape="diamond", arms=4, names=("DIAMANTE", "OURO"), emo=("\U0001F48E", "\U0001FA99"),
                    col=((250, 200, 90), (40, 190, 245)), snd=("ice", "coin"), bg=(28, 16, 12)),
}


class Arena:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pts = np.array(cfg["pts"] if "pts" in cfg else shape_pts(cfg["shape"]), np.float64)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [(self.pts * SH).astype(np.int32)], 255, cv2.LINE_8, 4)
        self.mask = mask
        din = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        sd = (din - dout).astype(np.float32)
        sds = cv2.GaussianBlur(sd, (0, 0), 1.5)
        gy, gx = np.gradient(sds)
        nn = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        self.sd, self.gx, self.gy = sd, (gx / nn).astype(np.float32), (gy / nn).astype(np.float32)
        iy, ix = np.unravel_index(np.argmax(din), din.shape)
        if cfg["shape"] in ("square", "rect", "circle", "octagon", "diamond"):
            ix, iy = int(round(self.pts[:, 0].mean())), int(round(self.pts[:, 1].mean()))
        self.C = np.array([float(ix), float(iy)])
        self.r_in = float(din[iy, ix])
        self.area = float((mask > 0).sum())
        self.bottom = float(self.pts[:, 1].max())
        # tamanho do emoji -> ~560 cabem
        self.RB = cfg.get("RB") or round(math.sqrt(0.78 * self.area / (2 * math.sqrt(3) * 560)), 1)
        self.CAP = int(0.80 * 0.78 * self.area / (2 * math.sqrt(3) * self.RB ** 2))
        # caixa do rotor
        r = self.r_in - 14
        bxs = cfg.get("box", 1.0)
        self.D = 0.27 * self.r_in * bxs
        self.Wb = 0.36 * self.r_in * bxs
        self.PK = 13.0
        self.R2 = math.sqrt(r * r - self.D ** 2) - self.PK / 2
        self.R1 = self.R2 - self.Wb
        self.GEAR = 30.0
        if self.R1 < self.GEAR + 22:
            self.R1 = self.GEAR + 22; self.Wb = self.R2 - self.R1
        self.Rm = (self.R1 + self.R2) / 2
        # contornos (normais p/ tremor)
        P = self.pts
        n = len(P)
        nrm = []
        for i in range(n):
            a, b = P[i - 1], P[(i + 1) % n]
            tvec = b - a; tvec /= np.linalg.norm(tvec) + 1e-9
            nv = np.array([tvec[1], -tvec[0]])
            mid = P[i] + nv * 3
            if self.inside(mid[None])[0]:
                nv = -nv
            nrm.append(nv)
        self.nrm = np.array(nrm)

    def inside(self, p):
        x = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        return self.mask[y, x] > 0

    def sample(self, p, s=1.0):
        q = self.C + (p - self.C) / s
        x = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        return self.sd[y, x] * s, self.gx[y, x], self.gy[y, x]

    def outline(self, s=1.0, wob=0.0, t=0.0):
        P = self.C + (self.pts - self.C) * s
        if wob <= 0:
            return P
        n = len(P)
        k = np.arange(n)
        bump = wob * (0.6 * np.sin(k * 2 * math.pi * 5 / n + t * 23) + 0.4 * np.sin(k * 2 * math.pi * 9 / n - t * 31))
        bump = np.abs(bump) * 0.6 + bump * 0.4
        return P + self.nrm * bump[:, None]

    def dense_outline(self, s=1.0, wob=0.0, t=0.0):
        # subdivide lados longos para tremor bonito
        P = self.C + (self.pts - self.C) * s
        out, nr = [], []
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            L = np.linalg.norm(b - a)
            m = max(1, int(L / 12))
            tv = (b - a) / (L + 1e-9); nv = np.array([tv[1], -tv[0]])
            if not self.inside(((a + b) / 2 - nv * 3)[None])[0]:
                nv = -nv
            for j in range(m):
                out.append(a + (b - a) * j / m); nr.append(nv)
        out = np.array(out); nr = np.array(nr)
        if wob > 0:
            k = np.arange(len(out))
            nn = len(out)
            bump = wob * (0.7 + 0.3 * np.sin(k * 0.35 + t * 25)) * (0.6 + 0.4 * np.sin(k * 2 * math.pi * 6 / nn + t * 9))
            out = out + nr * bump[:, None]
        return out


# ======================= mecanismo =======================
def cycle_phase(t):
    """(k, u) — índice do ciclo e tempo dentro do ciclo"""
    if t < T0:
        return -1, t - T0 + CYC
    k = int((t - T0) // CYC)
    return k, (t - T0) - k * CYC


def door_beta(u):
    if u < 0.35:
        x = u / 0.35; return (x * x * (3 - 2 * x)) * math.pi / 2
    if u < 0.75:
        return math.pi / 2
    if u < 1.1:
        x = (u - 0.75) / 0.35; return (1 - x * x * (3 - 2 * x)) * math.pi / 2
    return 0.0


def hammer_frac(u):
    if 0.35 <= u < 0.70:
        x = (u - 0.35) / 0.35; return x ** 2.2
    if 0.70 <= u < 0.78:
        return 1.0
    if 0.78 <= u < 1.05:
        x = (u - 0.78) / 0.27; return 1 - x * x * (3 - 2 * x)
    return 0.0


def mech_at(ar, t):
    th = OMEGA * t
    k, u = cycle_phase(t)
    beta = door_beta(u)
    hf = hammer_frac(u)
    PK = ar.PK; D = ar.D; Wb = ar.Wb; RB = ar.RB
    hx_max = D - PK / 2 - 7.5 - (2 * RB - 5)
    hx = 6 + (hx_max - 6) * hf
    boxes = []
    n = ar.cfg["arms"]
    for i in range(n):
        a = th + 2 * math.pi * i / n
        ue = np.array([math.cos(a), math.sin(a)])
        f = np.array([-ue[1], ue[0]])               # frente (sentido do giro)
        B = ar.C + ue * ar.Rm                        # meio da parede de trás
        po, pi_ = B + ue * Wb / 2, B - ue * Wb / 2
        ho, hi = po + f * D, pi_ + f * D              # dobradiças das portas
        dl = Wb / 2 - 1.5
        do = ho + dl * (math.cos(beta) * (-f) + math.sin(beta) * (-ue))
        di = hi + dl * (math.cos(beta) * (-f) + math.sin(beta) * (ue))
        hm = B + f * hx
        segs = [(pi_, po, PK), (po, ho, PK), (pi_, hi, PK), (ho, do, 11.0), (hi, di, 11.0),
                (hm - ue * (Wb / 2 - PK / 2 - 3), hm + ue * (Wb / 2 - PK / 2 - 3), 15.0)]
        boxes.append(dict(i=i, ue=ue, f=f, B=B, po=po, pi=pi_, ho=ho, hi=hi, do=do, di=di, hm=hm, hx=hx, segs=segs))
    return th, k, u, beta, boxes


def push_capsule(pos, mask, a, b, th, RB):
    ab = b - a
    L2 = ab @ ab
    if L2 < 1e-6:
        return
    s = np.clip(((pos - a) @ ab) / L2, 0, 1)
    cp = a + s[:, None] * ab
    d = pos - cp
    dist = np.linalg.norm(d, axis=1)
    lim = RB + th / 2
    hit = mask & (dist < lim)
    if hit.any():
        pos[hit] = cp[hit] + d[hit] / np.maximum(dist[hit], 1e-4)[:, None] * lim


# ======================= simulação =======================
LEVELS = [1.012, 1.035, 1.058, 1.08]
T_FULL = float(os.environ.get("T_FULL", "33"))
GROW_K = float(os.environ.get("GROW_K", "0.15"))
RUBBER = float(os.environ.get("RUBBER", "0.6"))


def simulate(ar, seed, mult, keep=True, swirl=-0.35):
    rng = np.random.default_rng(seed)
    RB = ar.RB; C = ar.C
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2)); ang = np.zeros(0)
    team = np.zeros(0, np.int8); ghost = np.zeros(0, bool); trap = np.zeros(0, np.int8)
    frames, audio, pops, leads, counts = [], [], [], [], []
    st = dict(s=1.0, exploded=False, t_over=None, t_expl=None, lvl=0, t_lastpop=None, leader=None, winner=None, final=None)
    frags, spawn_q, done = [], [], set()
    NF = int(DUR * FPS)

    def add(n, p0, v0, tm):
        nonlocal pos, prev, ang, team, ghost, trap
        p = p0 + rng.normal(0, 4, (n, 2))
        v = v0 + rng.normal(0, 60, (n, 2))
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        ang = np.concatenate([ang, rng.uniform(-0.5, 0.5, n)])
        team = np.concatenate([team, np.full(n, tm, np.int8)])
        ghost = np.concatenate([ghost, np.zeros(n, bool)])
        trap = np.concatenate([trap, np.zeros(n, np.int8)])

    # um de cada time dentro das duas primeiras caixas (no instante em que fecham)
    _, _, _, _, bx0 = mech_at(ar, T0 + 0.35)
    for tm, i in ((0, 0), (1, 1 if ar.cfg["arms"] > 2 else 1)):
        b = bx0[i]
        add(1, b["B"] + b["f"] * (ar.D * 0.62), np.zeros(2), tm)
    pos[:] = [bx0[i]["B"] + bx0[i]["f"] * (ar.D * 0.62) for i in (0, 1)]
    prev = pos.copy(); ghost[:] = True
    audio.append((0.15, "gong", 0.8))
    t_tick, tick_i = 0.5, 0

    for fi in range(NF):
        t = fi / FPS
        ex = st["exploded"]
        while t_tick <= t:
            if not ex:
                audio.append((t_tick, "tick" if tick_i % 2 == 0 else "tock", 0.1)); tick_i += 1
            t_tick += (0.07 if st["lvl"] >= 3 else 0.5 / (1 + st["lvl"] * 1.3)) if (st["t_over"] is not None and not ex) else 0.5
        th, k, u, beta, boxes = mech_at(ar, t)

        if not ex and k >= 0:
            tc = T0 + k * CYC
            if ("trap", k) not in done and u >= 0.35:
                done.add(("trap", k)); ghost[:] = False
                audio.append((tc + 0.35, "clack", 0.3))
                for b in boxes:
                    d = pos - b["B"]
                    x = d @ b["f"]; y = d @ b["ue"]
                    ins = (x > b["hx"] + 7) & (x < ar.D - 3) & (np.abs(y) < ar.Wb / 2 - ar.PK / 2) & (trap == 0)
                    trap[ins] = b["i"] + 1
            if ("press", k) not in done and u >= 0.70:
                done.add(("press", k))
                audio.append((tc + 0.70, "press", 0.4))
                if (trap != 0).any():
                    grow = st["t_over"] is not None
                    if grow:
                        st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                        audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                    tot_tr = int((trap != 0).sum())
                    target = ar.CAP * math.exp(GROW_K * (t + CYC - T_FULL))
                    deficit = max(0.0, target - len(pos))
                    for b in boxes:
                        sel = trap == b["i"] + 1
                        nt = int(sel.sum())
                        if not nt:
                            continue
                        na = int((sel & (team == 0)).sum()); nd = nt - na
                        xs = (pos[sel] - b["B"]) @ b["f"]
                        w = int(team[sel][int(np.argmin(xs))])     # quem encosta no martelo leva tudo
                        km = ~sel
                        pos, prev, ang, team, ghost, trap = pos[km], prev[km], ang[km], team[km], ghost[km], trap[km]
                        cur = [int((team == 0).sum()), int((team == 1).sum())]
                        rb = 1.0
                        if cur[0] != cur[1]:
                            trailing = 0 if cur[0] < cur[1] else 1
                            gap = abs(cur[0] - cur[1]) / max(1, sum(cur))
                            rb = (1 + RUBBER * gap * 4) if w == trailing else max(0.35, 1 - RUBBER * gap * 3)
                        n = int(round((nt * mult + deficit * nt / tot_tr) * rb))
                        if grow:
                            capn = int(ar.CAP / 0.80 * LEVELS[st["lvl"]] ** 2)
                            n = max(1, min(n, (capn - len(pos)) // ar.cfg["arms"] + 2))
                        else:
                            n = max(1, min(n, ar.CAP + 15 - len(pos)))
                        if min(na, nd) > 0:
                            audio.append((t + 0.02, "steal", 0.6))
                        pulses = max(1, min(8, (n + 2) // 3))
                        base, rest = divmod(n, pulses)
                        for p in range(pulses):
                            cnt = base + (1 if p < rest else 0)
                            side = 1 if p % 2 == 0 else -1
                            spawn_q.append([t + 0.06 + p * 0.07, cnt, b["i"], side, w])
                        pops.append((t, (b["B"] + b["f"] * ar.D * 0.5).tolist(), n, w, min(na, nd) > 0))
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, bi, side, tm = spawn_q.pop(0)
            if not st["exploded"] and n > 0:
                b = boxes[bi]
                p0 = b["B"] + b["f"] * ar.D * 0.5 + b["ue"] * side * (ar.Wb / 2 + RB + 8)
                v0 = b["ue"] * side * 230 + b["f"] * 60
                if not ar.inside(p0[None])[0]:
                    p0 = b["B"] + b["f"] * ar.D * 0.5 - b["ue"] * (ar.Wb / 2 + RB + 8); v0 = -b["ue"] * 230
                add(n, p0, v0, tm)
                audio.append((ts, "t%d" % tm, 0.5))

        na = int((team == 0).sum()); nd = int((team == 1).sum())
        if not ex:
            if na != nd and na + nd >= 12:
                ld = 0 if na > nd else 1
                if st["leader"] is not None and ld != st["leader"]:
                    leads.append((t, ld)); audio.append((t, "lead", 0.55))
                st["leader"] = ld
            elif st["leader"] is None and na != nd:
                st["leader"] = 0 if na > nd else 1

        if not ex and st["t_over"] is None and len(pos) >= ar.CAP:
            st["t_over"] = t; audio.append((t, "rumble", 0.5))
        if not ex and st["t_over"] is not None:
            st["s"] += (LEVELS[st["lvl"]] - st["s"]) * 0.06
            if st["lvl"] >= 3 and t >= st["t_lastpop"] + 1.0:
                st.update(exploded=True, t_expl=t); ex = True
                st["final"] = (na, nd); st["winner"] = 0 if na >= nd else 1
                audio.append((t, "boom", 1.0)); audio.append((t + 0.9, "fanfare", 0.9)); audio.append((t + 0.9, "crowd", 0.7))
                d = pos - C
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                spd = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / ar.r_in)
                prev = pos - (d / dist * spd + rng.normal(0, 120, pos.shape)) * DT
                P = ar.outline(st["s"])
                n = len(P)
                step = max(1, n // 36)
                for i in range(0, n, step):
                    a, b2 = P[i], P[(i + step) % n]
                    m = (a + b2) / 2; dv = (m - C) / (np.linalg.norm(m - C) + 1e-6)
                    frags.append(dict(a=a - m, b=(b2 - m) * 0.9, c=m, rot=0.0, v=dv * rng.uniform(500, 1100), w=rng.uniform(-6, 6)))

        grav = np.array([0.0, 900.0]) if ex else np.zeros(2)
        for s in range(SUB):
            if len(pos) == 0:
                break
            ts = t + s * DT
            v = (pos - prev) / DT * 0.9993
            if not ex:
                rel = pos - C
                vt = np.stack([-rel[:, 1], rel[:, 0]], 1) * swirl
                v += (vt - v) * min(1.0, 0.7 * DT)
                if ar.cfg.get("pull", 90):
                    rr = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-6
                    v -= (rel / rr) * np.clip((rr - ar.Rm) / ar.r_in, -1, 1) * ar.cfg.get("pull", 90) * DT
                v[ghost] = 0
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            cap = 2500 if ex else 650
            v = np.where(spd > cap, v / np.maximum(spd, 1e-6) * cap, v)
            prev = pos; pos = pos + v * DT + grav * DT * DT
            ang = ang + v[:, 0] * DT / RB * 0.35
            _, _, _, _, bxs = mech_at(ar, ts)
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
                    solid = ~ghost
                    free = solid & (trap == 0)
                    for b in bxs:
                        for a, b2, thk in b["segs"]:
                            push_capsule(pos, free, a, b2, thk, RB)
                        sel = trap == b["i"] + 1
                        if sel.any():
                            d = pos[sel] - b["B"]
                            x = np.clip(d @ b["f"], b["hx"] + 7.5 + RB * 0.55, ar.D - ar.PK / 2 - RB * 0.55)
                            lim = ar.Wb / 2 - ar.PK / 2 - RB
                            y = np.clip(d @ b["ue"], -lim, lim)
                            pos[sel] = b["B"] + x[:, None] * b["f"] + y[:, None] * b["ue"]
                    dg = pos - C
                    dd = np.linalg.norm(dg, axis=1)
                    hg = solid & (dd < RB + ar.GEAR)
                    if hg.any():
                        pos[hg] = C + dg[hg] / np.maximum(dd[hg], 1e-4)[:, None] * (RB + ar.GEAR)
                    sd, gx, gy = ar.sample(pos, st["s"])
                    o = sd < RB
                    if o.any():
                        g = np.stack([gx[o], gy[o]], 1)
                        pos[o] += g * (RB - sd[o])[:, None]
                        vv = pos[o] - prev[o]
                        vn = (vv * g).sum(1)
                        prev[o] = pos[o] - (vv - np.minimum(vn, 0)[:, None] * g * 1.85)
        if ex:
            kp = (pos[:, 0] > -60) & (pos[:, 0] < W + 60) & (pos[:, 1] < H + 60) & (pos[:, 1] > -600)
            pos, prev, ang, team, ghost, trap = pos[kp], prev[kp], ang[kp], team[kp], ghost[kp], trap[kp]
            for fr in frags:
                fr["c"] = fr["c"] + fr["v"] / FPS; fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS
                fr["rot"] += fr["w"] / FPS
        counts.append((na, nd) if not ex else st["final"])
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), ang=ang.copy(), team=team.copy(), s=st["s"], exploded=ex,
                               t_over=st["t_over"], lvl=st["lvl"], t_lastpop=st["t_lastpop"], t_expl=st["t_expl"],
                               frags=[dict(f) for f in frags], counts=counts[-1]))
        if ex and t > st["t_expl"] + 4.6:
            break
    return frames, audio, pops, leads, st, counts


# ======================= desenho =======================
def emoji_img(ch):
    im = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((12, 12), ch, font=EMOJI_FONT, embedded_color=True)
    im = im.crop(im.getbbox())
    s = max(im.size)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0)); sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return sq


def to_pre(im):
    a = np.asarray(im).astype(np.float32)
    al = a[..., 3:4] / 255
    return (a[..., [2, 1, 0]] * al).astype(np.float32), (1 - al).astype(np.float32)


_cache = {}


def P(p):
    return (int(round(p[0] * SH)), int(round(p[1] * SH)))


def blit(f, pre, inv, x0, y0, alpha=1.0):
    h, w = pre.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    if alpha >= 1:
        f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * inv[sl] + pre[sl]
    else:
        f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - (1 - inv[sl]) * alpha) + pre[sl] * alpha


def emoji_scaled(img, d):
    key = ("e", id(img), d)
    if key not in _cache:
        _cache[key] = to_pre(img.resize((d, d), Image.LANCZOS))
    return _cache[key]


def text_img(txt, size, color, stroke, stroke_col):
    key = ("t", txt, size, color, stroke, stroke_col)
    if key not in _cache:
        font = ImageFont.truetype(FONT_B, size)
        bb = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), txt, font=font, stroke_width=stroke)
        im = Image.new("RGBA", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), txt, font=font, fill=(color[2], color[1], color[0], 255),
                                stroke_width=stroke, stroke_fill=(stroke_col[2], stroke_col[1], stroke_col[0], 255))
        _cache[key] = to_pre(im)
    return _cache[key]


def put_text(f, txt, size, cx, cy, color=(255, 255, 255), stroke=0, alpha=1.0, stroke_col=(0, 0, 0), scale=1.0, maxw=None):
    pre, inv = text_img(txt, size, color, stroke, stroke_col)
    h, w = pre.shape[:2]
    if maxw and w * scale > maxw:
        scale = maxw / w
    if abs(scale - 1) > 0.02:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pre = cv2.resize(pre, (nw, nh)); inv = cv2.resize(inv[..., 0], (nw, nh))[..., None]
    h, w = pre.shape[:2]
    blit(f, pre, inv, int(cx - w / 2), int(cy - h / 2), alpha)


def make_glow(r):
    s = int(r * 2)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    return np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 2


_GL = {}


def add_glow(f, center, color, k, r=120):
    if r not in _GL:
        _GL[r] = make_glow(r)
    g = _GL[r]; s = g.shape[0]
    x0, y0 = int(center[0] - s / 2), int(center[1] - s / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if xa >= xb or ya >= yb or k <= 0.01:
        return
    f[ya:yb, xa:xb] += g[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * k * np.array(color, np.float32)


def gear_poly(center, a0, r, teeth):
    pts = []
    for i in range(teeth * 4):
        a = a0 + i * (2 * math.pi / (teeth * 4))
        rr = r if (i % 4) in (0, 1) else r * 0.8
        pts.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a)))
    return (np.array(pts) * SH).astype(np.int32)


def draw_rod(img, a, b, th, base=(170, 165, 170)):
    d = b - a; L = float(np.linalg.norm(d))
    if L < 0.5:
        return
    u = d / L; n = np.array([-u[1], u[0]]); h = th / 2
    cv2.line(img, P(a), P(b), (55, 50, 55), int(th + 3), cv2.LINE_AA, 4)
    cv2.line(img, P(a), P(b), base, int(th), cv2.LINE_AA, 4)
    cv2.line(img, P(a + n * (h - 3)), P(b + n * (h - 3)), tuple(min(255, c + 55) for c in base), 2, cv2.LINE_AA, 4)
    cv2.line(img, P(a - n * (h - 3)), P(b - n * (h - 3)), tuple(int(c * 0.62) for c in base), 2, cv2.LINE_AA, 4)


def ease_out_back(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Conf:
    def __init__(self, seed, t0, cols):
        r = np.random.default_rng(seed)
        n = 170
        self.t0 = t0
        self.p0 = np.stack([r.uniform(0, W, n), r.uniform(-500, -20, n)], 1)
        self.v = np.stack([r.normal(0, 60, n), r.uniform(260, 520, n)], 1)
        pal = [[60, 60, 255], [60, 220, 255], [90, 230, 90], [255, 255, 255], list(cols[0]), list(cols[1])]
        self.col = [tuple(int(c) for c in pal[r.integers(0, len(pal))]) for _ in range(n)]
        self.ph = r.uniform(0, 6, n); self.w = r.uniform(4, 10, n)

    def draw(self, img, t):
        dt = t - self.t0
        if dt < 0:
            return
        for i in range(len(self.p0)):
            x = self.p0[i, 0] + self.v[i, 0] * dt + 20 * math.sin(dt * 3 + self.ph[i])
            y = self.p0[i, 1] + self.v[i, 1] * dt
            if -20 < y < H + 20:
                a = dt * self.w[i] + self.ph[i]
                box = cv2.boxPoints(((x, y), (2 * (7 * abs(math.cos(a)) + 1), 8), math.degrees(a * 0.7)))
                cv2.fillPoly(img, [box.astype(np.int32)], self.col[i], cv2.LINE_AA)


class Renderer:
    def __init__(self, ar):
        _cache.clear()
        self.ar = ar
        cfg = ar.cfg
        self.EMO = [emoji_img(c) for c in cfg["emo"]]
        self.TC = cfg["col"]
        self.NROT = 24
        d = int(round(ar.RB * 2.3))
        self.SD = d; self.HALF = d // 2
        self.SPR = []
        for im in self.EMO:
            big = im.resize((d * 4, d * 4), Image.LANCZOS)
            self.SPR.append([to_pre(big.rotate(-360 * r / self.NROT, resample=Image.BICUBIC).resize((d, d), Image.LANCZOS))
                             for r in range(self.NROT)])
        self.CROWN = emoji_img("\U0001F451"); self.FINGER = emoji_img("\U0001F447"); self.TROPHY = emoji_img("\U0001F3C6")
        # fundo + piso
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dd = np.sqrt((xx - ar.C[0]) ** 2 + (yy - ar.C[1]) ** 2) / 750
        g = np.clip(1 - dd, 0, 1) ** 1.5
        bg = np.zeros((H, W, 3), np.float32); bg[:] = (8, 5, 8)
        bg += g[..., None] * np.array(cfg["bg"], np.float32) * 1.6
        self.BG = bg
        fl = np.zeros((H, W), np.float32)
        fl[ar.mask > 0] = 1
        self.floor = cv2.GaussianBlur(fl, (0, 0), 1.0)[..., None]
        self.floor_col = np.array(cfg["bg"], np.float32) * 1.1 + 10
        self.disp = 0.5

    def render(self, fr, pops, leads, st, srng, conf):
        ar = self.ar; C = ar.C; t = fr["t"]; ex = fr["exploded"]; cfg = ar.cfg
        f = self.BG.copy()
        if not ex:
            s = fr["s"]
            if abs(s - 1) > 0.002:
                M = np.float32([[s, 0, C[0] * (1 - s)], [0, s, C[1] * (1 - s)]])
                fl = cv2.warpAffine(self.floor, M, (W, H))[..., None]
            else:
                fl = self.floor
            f = f * (1 - fl * 0.6) + fl * 0.6 * self.floor_col
        for (t0, gc, n, tm, steal) in pops:
            if t0 - 0.02 <= t < t0 + 0.9:
                k = 1.0 if t < t0 + 0.12 else max(0.0, 1 - (t - t0 - 0.12) / 0.78)
                add_glow(f, gc, self.TC[tm], k * (1.0 if steal else 0.65))
        ip = np.round(fr["pos"]).astype(np.int32)
        ri = (np.round(fr["ang"] / (2 * math.pi) * self.NROT).astype(int)) % self.NROT
        tm_ = fr["team"]; SD = self.SD; HALF = self.HALF
        for k in range(len(ip)):
            pre, inv = self.SPR[tm_[k]][ri[k]]
            x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
            if 0 <= x0 and 0 <= y0 and x0 + SD <= W and y0 + SD <= H:
                reg = f[y0:y0 + SD, x0:x0 + SD]; reg *= inv; reg += pre
            else:
                blit(f, pre, inv, x0, y0)
        img = np.clip(f, 0, 255).astype(np.uint8)

        na, nd = fr["counts"]
        tot = max(1, na + nd)
        self.disp += (na / tot - self.disp) * 0.15
        lead_col = self.TC[0] if na >= nd else self.TC[1]

        wob = 0.0
        if fr["t_over"] is not None and not ex:
            lv = fr["lvl"]
            kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.6) if fr["t_lastpop"] is not None else 0.0
            kick0 = max(0.0, 1 - (t - fr["t_over"]) / 0.8)
            wob = [3, 6, 10, 15][lv] + 10 * kick + 8 * kick0
            if lv >= 3:
                wob += 10 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
        if not ex:
            pts = (ar.dense_outline(fr["s"], wob, t) * SH).astype(np.int32)
            glow = np.zeros_like(img)
            cv2.polylines(glow, [pts], True, lead_col, 14, cv2.LINE_AA, 4)
            glow = cv2.GaussianBlur(glow, (0, 0), 9)
            img = cv2.add(img, (glow * 0.85).astype(np.uint8))
            cv2.polylines(img, [pts], True, (245, 240, 250), 4, cv2.LINE_AA, 4)
            if cfg.get("posts"):
                for p in ar.outline(fr["s"]):
                    cv2.circle(img, P(p), 9 * SH, (25, 22, 25), -1, cv2.LINE_AA, 4)
                    cv2.circle(img, P(p), 9 * SH, (90, 85, 90), 2, cv2.LINE_AA, 4)
                    cv2.circle(img, P(p - np.array([2.5, 2.5])), 3 * SH, (150, 145, 150), -1, cv2.LINE_AA, 4)
        for fg in fr["frags"]:
            cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"])
            R = np.array([[cr, -sr], [sr, cr]])
            cv2.line(img, P(fg["c"] + R @ fg["a"]), P(fg["c"] + R @ fg["b"]), (245, 240, 250), 4, cv2.LINE_AA, 4)

        ta = 1.0 if not ex else max(0.0, 1 - (t - fr["t_expl"]) / 0.25)
        if ta > 0:
            th, k, u, beta, boxes = mech_at(ar, t)
            ov = img.copy()
            for b in boxes:
                cv2.line(ov, P(C + np.array([5.0, 7.0])), P(b["B"] + np.array([5.0, 7.0])), (6, 4, 6), 19, cv2.LINE_AA, 4)
            for b in boxes:
                draw_rod(ov, C, b["B"], 14)
                # martelo
                draw_rod(ov, b["B"], b["hm"], 7, base=(120, 115, 120))
                a_, b_, th_ = b["segs"][5]
                draw_rod(ov, a_, b_, th_, base=(70, 180, 235))
                # caixa
                for a_, b_, th_ in b["segs"][:3]:
                    draw_rod(ov, a_, b_, th_)
                for a_, b_, th_ in b["segs"][3:5]:
                    draw_rod(ov, a_, b_, th_, base=(60, 160, 215))
                for cc in (b["po"], b["pi"], b["ho"], b["hi"]):
                    cv2.circle(ov, P(cc), int(ar.PK / 2 * SH), (170, 165, 170), -1, cv2.LINE_AA, 4)
                    cv2.circle(ov, P(cc), int(2.5 * SH), (60, 55, 60), -1, cv2.LINE_AA, 4)
            gp = gear_poly(C, th, ar.GEAR, 14)
            cv2.fillPoly(ov, [gp], (70, 175, 215), cv2.LINE_AA, 4)
            cv2.polylines(ov, [gp], True, (30, 90, 120), 1, cv2.LINE_AA, 4)
            cv2.circle(ov, P(C), int(ar.GEAR * 0.6 * SH), (40, 120, 160), 2, cv2.LINE_AA, 4)
            cv2.circle(ov, P(C), int(10 * SH), (200, 195, 200), -1, cv2.LINE_AA, 4)
            img = ov if ta >= 1 else cv2.addWeighted(ov, ta, img, 1 - ta, 0)

        f = img.astype(np.float32)
        for (t0, gc, n, tm, steal) in pops:
            dtt = t - t0
            if 0 <= dtt < 0.9:
                gc = np.array(gc)
                p = gc + (gc - C) / max(1e-3, np.linalg.norm(gc - C)) * (-20 - dtt * 40) + np.array([0, -dtt * 30])
                al = min(1.0, (1 - dtt / 0.9) * 1.6)
                put_text(f, f"+{n}", 28 if not steal else 34, p[0], p[1], self.TC[tm], stroke=4, alpha=al, stroke_col=(20, 10, 15))

        # placar
        n0, n1 = cfg["names"]
        put_text(f, f"{n0}  vs  {n1}", 44, W / 2, 60, (255, 255, 255), stroke=3, stroke_col=(40, 20, 30), maxw=660)
        bx0, bx1, by = 70, 650, 214
        bump = [1.0, 1.0]
        for (t0, gc, n, tm, steal) in pops:
            if 0 <= t - t0 < 0.25:
                bump[tm] = 1.18
        blit(f, *emoji_scaled(self.EMO[0], 64), 40, 112); blit(f, *emoji_scaled(self.EMO[1], 64), W - 104, 112)
        put_text(f, str(na), 52, 205, 146, self.TC[0], stroke=4, stroke_col=(30, 15, 20), scale=bump[0])
        put_text(f, str(nd), 52, W - 205, 146, self.TC[1], stroke=4, stroke_col=(30, 15, 20), scale=bump[1])
        img = np.clip(f, 0, 255).astype(np.uint8)
        xs = int(bx0 + (bx1 - bx0) * self.disp)
        cv2.rectangle(img, (bx0 - 3, by - 17), (bx1 + 3, by + 17), (40, 25, 35), -1, cv2.LINE_AA)
        cv2.rectangle(img, (bx0, by - 14), (xs, by + 14), self.TC[0], -1, cv2.LINE_AA)
        cv2.rectangle(img, (xs, by - 14), (bx1, by + 14), self.TC[1], -1, cv2.LINE_AA)
        hl = img[by - 13:by - 5, bx0:bx1].astype(np.float32)
        img[by - 13:by - 5, bx0:bx1] = np.clip(hl + 45, 0, 255).astype(np.uint8)
        cv2.circle(img, (xs, by), 19, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (xs, by), 19, (40, 25, 35), 2, cv2.LINE_AA)
        f = img.astype(np.float32)
        put_text(f, f"{round(100 * na / tot)}%", 20, bx0 + 40, by, (255, 255, 255), stroke=2, stroke_col=(40, 30, 30))
        put_text(f, f"{round(100 * nd / tot)}%", 20, bx1 - 40, by, (255, 255, 255), stroke=2, stroke_col=(40, 30, 30))
        put_text(f, "VS", 18, xs, by, (40, 25, 35))

        fy = max(1075, ar.bottom + 55)
        recent = [(t0, ld) for (t0, ld) in leads if 0 <= t - t0 < 1.1]
        if recent and not ex:
            t0, ld = recent[-1]
            dtt = t - t0
            sc = ease_out_back(min(1.0, dtt / 0.25))
            al = 1.0 if dtt < 0.8 else max(0.0, 1 - (dtt - 0.8) / 0.3)
            if sc > 0.15:
                put_text(f, "VIROU!", 64, W / 2, fy + 10, self.TC[ld], stroke=6, stroke_col=(255, 255, 255), alpha=al, scale=sc)
        elif not ex:
            if fr["t_over"] is not None:
                put_text(f, "VAI EXPLODIR!", 50, W / 2, fy + 10, (60, 60, 255), stroke=5, stroke_col=(255, 255, 255), scale=1 + 0.08 * math.sin(t * 18))
            else:
                put_text(f, "Quem vence?", 44, W / 2 - 30, fy, (255, 255, 255), stroke=3, stroke_col=(40, 20, 30), scale=1 + 0.04 * math.sin(t * 5))
                blit(f, *emoji_scaled(self.FINGER, 52), int(W / 2 + 130), int(fy - 26 + 6 * math.sin(t * 6)))
                put_text(f, "comenta seu palpite", 26, W / 2, fy + 58, (200, 190, 210))
        img = np.clip(f, 0, 255).astype(np.uint8)

        if ex:
            te = t - fr["t_expl"]
            fl = max(0.0, 0.85 * (1 - te / 0.35))
            if fl > 0:
                img = cv2.addWeighted(img, 1 - fl, np.full_like(img, 255), fl, 0)
            if te > 0.7:
                k = min(1.0, (te - 0.7) / 0.4)
                img = (img.astype(np.float32) * (1 - 0.65 * k)).astype(np.uint8)
                conf.draw(img, t)
                f = img.astype(np.float32)
                wv = st["winner"]
                sc = ease_out_back(min(1.0, (te - 0.8) / 0.45)) if te > 0.8 else 0
                if sc > 0.02:
                    d = int(250 * sc * (1 + 0.04 * math.sin(te * 6)))
                    add_glow(f, (W / 2, 640), self.TC[wv], 1.4 * min(1, sc), 220)
                    blit(f, *emoji_scaled(self.EMO[wv], max(4, d)), int(W / 2 - d / 2), int(650 - d / 2))
                    if te > 1.2:
                        cs = ease_out_back(min(1.0, (te - 1.2) / 0.35))
                        cd = int(150 * max(0.05, cs))
                        blit(f, *emoji_scaled(self.CROWN, cd), int(W / 2 - cd / 2), int(650 - d / 2 - cd * 0.75 + 10 * math.sin(te * 4)))
                    put_text(f, f"{cfg['names'][wv]} VENCEU!", 80, W / 2, 920, self.TC[wv], stroke=7, stroke_col=(255, 255, 255), scale=max(0.05, sc), maxw=680)
                    fa, fd = st["final"]
                    put_text(f, f"{fa}  x  {fd}", 50, W / 2, 1025, (255, 255, 255), stroke=4, stroke_col=(40, 20, 30), alpha=min(1, sc))
                    if te > 1.6:
                        put_text(f, "Você acertou?", 40, W / 2, 1110, (255, 255, 255), stroke=3, stroke_col=(40, 20, 30), alpha=min(1, (te - 1.6) / 0.3))
                img = np.clip(f, 0, 255).astype(np.uint8)
            shake = max(0.0, 18 * (1 - te / 0.7))
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


# ======================= áudio =======================
SR = 44100


def make_audio(events, dur, snd, path, seed=1):
    rng = np.random.default_rng(seed)
    N = int(SR * dur)
    out = np.zeros((N, 2), np.float32)

    def env(n, tau): return np.exp(-np.arange(n) / SR / tau)
    def lp(x, a): return lfilter([a], [1, a - 1], x)
    def tt(n): return np.arange(n) / SR

    def tone(f, dur, tau, harm=((1, 1.0),)):
        n = int(SR * dur); t = tt(n)
        return sum(a * np.sin(2 * np.pi * f * h * t) for h, a in harm) * env(n, tau) * np.minimum(1, t / 0.002)

    def glide(f0, f1, dur, tau, vib=0.0):
        n = int(SR * dur); t = tt(n)
        fr = f0 + (f1 - f0) * t / dur
        fr = fr * (1 + vib * np.sin(2 * np.pi * 6 * t))
        return np.sin(2 * np.pi * np.cumsum(fr) / SR) * env(n, tau) * np.minimum(1, t / 0.01)

    def team_sound(kind, i):
        if kind == "roar":
            n = int(SR * 0.4); t = tt(n); f0 = 85 + 10 * (i % 3)
            fr = f0 * (1.3 - 0.4 * t / 0.4)
            saw = 2 * ((np.cumsum(fr) / SR) % 1) - 1
            s = lp(saw + rng.normal(0, 0.6, n), 0.08) * np.minimum(1, t / 0.04) * env(n, 0.15)
            return s * (1 + 0.5 * np.sin(2 * np.pi * 28 * t))
        if kind == "howl":
            n = int(SR * 0.5); t = tt(n); b = 420 + 40 * (i % 4)
            fr = b + 260 * np.sin(np.pi * t / 0.5) ** 0.8
            return np.sin(2 * np.pi * np.cumsum(fr * (1 + 0.012 * np.sin(2 * np.pi * 7 * t))) / SR) * np.sin(np.pi * t / 0.5) ** 0.6 * 0.9
        if kind == "love":
            return glide(600 + 50 * (i % 4), 1300 + 80 * (i % 4), 0.18, 0.09) + tone(2093, 0.25, 0.08, ((1, 0.25),))[:int(SR * 0.18)]
        if kind == "angry":
            n = int(SR * 0.25); t = tt(n); f = 110 + 15 * (i % 3)
            sq = np.sign(np.sin(2 * np.pi * f * t)) + 0.5 * np.sign(np.sin(2 * np.pi * f * 1.06 * t))
            return lp(sq, 0.18) * env(n, 0.08)
        if kind == "sun":
            notes = [1046.5, 1318.5, 1568.0, 2093.0]
            return tone(notes[i % 4], 0.45, 0.14, ((1, 1), (2, 0.35), (3, 0.15)))
        if kind == "moon":
            notes = [392.0, 440.0, 523.3, 587.3]
            return tone(notes[i % 4], 0.6, 0.22, ((1, 1), (2.01, 0.3)))
        if kind == "kick":
            n = int(SR * 0.2); t = tt(n)
            return np.sin(2 * np.pi * (160 - 100 * t / 0.2) * t) * env(n, 0.05) + rng.normal(0, 1, n) * env(n, 0.004) * 0.8
        if kind == "bounce":
            n = int(SR * 0.3); t = tt(n)
            fr = 260 * np.exp(-t * 8) + 90
            return np.sin(2 * np.pi * np.cumsum(fr) / SR) * env(n, 0.09) * 1.2
        if kind == "bark":
            s = np.zeros(int(SR * 0.3))
            for d in (0.0, 0.13):
                n = int(SR * 0.11); t = tt(n)
                x = lp(rng.normal(0, 1, n), 0.2) * 2 + np.sin(2 * np.pi * (520 - 200 * t / 0.11) * t)
                x *= np.sin(np.pi * t / 0.11)
                i0 = int(d * SR); s[i0:i0 + n] += x[: len(s) - i0]
            return s
        if kind == "meow":
            n = int(SR * 0.35); t = tt(n)
            fr = 700 + 300 * np.sin(np.pi * t / 0.35) - 250 * t / 0.35
            s = np.sin(2 * np.pi * np.cumsum(fr) / SR) + 0.4 * np.sin(4 * np.pi * np.cumsum(fr) / SR)
            return s * np.sin(np.pi * t / 0.35) ** 0.7
        if kind == "pop":
            return glide(400 + 60 * (i % 4), 1100, 0.09, 0.04)
        if kind == "chomp":
            n = int(SR * 0.16); t = tt(n)
            return np.sin(2 * np.pi * 180 * t) * env(n, 0.04) + lp(rng.normal(0, 1, n), 0.3) * env(n, 0.03) * 1.5
        if kind == "ice":
            notes = [2093.0, 2637.0, 3136.0, 3520.0]
            f = notes[i % 4]; n = int(SR * 0.45); t = tt(n)
            return sum(a * np.sin(2 * np.pi * f * r * t) * env(n, 0.13 * dd) for r, a, dd in ((1, 1, 1), (2.76, 0.5, 0.5), (5.4, 0.3, 0.3)))
        if kind == "coin":
            n = int(SR * 0.4); t = tt(n); f0 = rng.uniform(2400, 3500)
            s = sum(a * np.sin(2 * np.pi * f0 * r * t) * env(n, 0.1 * dd) for r, a, dd in ((1, 1, 1), (2.32, 0.7, 0.8), (4.25, 0.45, 0.55)))
            s2 = np.zeros(n); d = int(0.035 * SR); s2[d:] = s[:-d] * 0.7
            return s + s2
        raise ValueError(kind)

    def click(f1, f2, tau):
        n = int(SR * 0.08); t = tt(n)
        return (0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)) * env(n, tau) + rng.normal(0, 1, n) * env(n, 0.0015) * 0.4

    def gong():
        s = np.zeros(int(SR * 1.6))
        for d in (0.0, 0.32, 0.64):
            x = tone(1250, 1.0, 0.35, ((1, 1), (2.4, 0.5), (3.9, 0.3), (5.6, 0.15)))
            i0 = int(d * SR); s[i0:i0 + len(x)] += x[: len(s) - i0]
        return s

    def lead():
        s = np.zeros(int(SR * 0.6))
        for d, fq in ((0, 1568.0), (0.12, 2093.0)):
            x = tone(fq, 0.45, 0.15, ((1, 1), (2, 0.3)))
            i0 = int(d * SR); s[i0:i0 + len(x)] += x[: len(s) - i0]
        return s

    def steal():
        n = int(SR * 0.3); t = tt(n)
        return np.sin(2 * np.pi * np.cumsum(1800 * np.exp(-t * 9) + 200) / SR) * env(n, 0.1)

    def press():
        n = int(SR * 0.25); t = tt(n)
        return np.sin(2 * np.pi * (140 - 60 * t / 0.25) * t) * env(n, 0.05) + (np.sin(2 * np.pi * 760 * t) + 0.5 * np.sin(2 * np.pi * 1290 * t)) * env(n, 0.02) * 0.4

    def clack():
        n = int(SR * 0.12); t = tt(n)
        return (np.sin(2 * np.pi * 1200 * t) + 0.6 * np.sin(2 * np.pi * 2200 * t)) * env(n, 0.015)

    def boom():
        n = int(SR * 2.4); t = tt(n); nz = rng.normal(0, 1, n)
        return lp(nz, 0.02) * 6 * env(n, 0.5) + lp(nz, 0.35) * env(n, 0.08) + np.sin(2 * np.pi * np.cumsum(70 - 40 * np.minimum(t, 1)) / SR) * env(n, 0.55) * 0.9

    def rumble():
        n = int(SR * 2.0); t = tt(n); e = (t / 2.0) ** 2
        return lp(rng.normal(0, 1, n), 0.01) * 8 * e + np.sin(2 * np.pi * 55 * t) * e * 0.4

    def fanfare():
        s = np.zeros(int(SR * 3.2))
        for d, fs in [(0.0, [523.3]), (0.14, [659.3]), (0.28, [784.0]), (0.42, [1046.5]), (0.75, [523.3, 659.3, 784.0, 1046.5])]:
            du = 0.3 if len(fs) == 1 else 2.2
            for fq in fs:
                n = int(SR * du); t = tt(n)
                x = lp(np.sign(np.sin(2 * np.pi * fq * t)) * 0.25 + np.sin(2 * np.pi * fq * t), 0.3) * np.minimum(1, t / 0.01) * env(n, 0.25 if len(fs) == 1 else 0.9)
                i0 = int(d * SR); s[i0:i0 + n] += x[: len(s) - i0]
        return s

    def crowd():
        n = int(SR * 3.0); t = tt(n)
        nz = lfilter([0.3], [1, -0.7], rng.normal(0, 1, n)) - lfilter([0.05], [1, -0.95], rng.normal(0, 1, n))
        e = np.minimum(1, t / 0.4) * np.exp(-np.maximum(0, t - 1.2) / 0.8)
        claps = np.zeros(n)
        for _ in range(140):
            i0 = int(rng.uniform(0, 2.4) * SR); m = 300
            if i0 + m < n:
                claps[i0:i0 + m] += rng.normal(0, 1, m) * env(m, 0.002) * rng.uniform(0.3, 1)
        return (nz * 2 + claps) * e

    base = {"tick": click(3100, 4700, 0.006), "tock": click(1750, 2650, 0.009), "press": press(), "clack": clack(),
            "steal": steal(), "lead": lead(), "boom": boom(), "rumble": rumble(), "fanfare": fanfare(), "gong": gong(), "crowd": crowd()}
    base = {k: v / np.abs(v).max() for k, v in base.items()}
    tvars = [[team_sound(snd[tm], i) for i in range(6)] for tm in (0, 1)]
    tvars = [[v / np.abs(v).max() for v in vs] for vs in tvars]
    gain = {"t0": 0.32, "t1": 0.32, "boom": 2.4, "fanfare": 1.9, "crowd": 1.2, "lead": 1.2, "steal": 0.8, "rumble": 1.3, "gong": 1.4}
    cnt = [0, 0]
    for (t, kind, vol) in events:
        if kind in ("t0", "t1"):
            tm = int(kind[1]); s = tvars[tm][cnt[tm] % 6]; cnt[tm] += 1
        else:
            s = base[kind]
        s = s * vol * gain.get(kind, 1.0)
        i0 = int(t * SR)
        if i0 >= N:
            continue
        seg = s[: N - i0]
        pan = {"t0": -0.35, "t1": 0.35, "tick": 0.15, "tock": -0.15}.get(kind, 0.0)
        out[i0:i0 + len(seg), 0] += seg * (1 - pan)
        out[i0:i0 + len(seg), 1] += seg * (1 + pan)
    for d, g in ((0.05, 0.22), (0.12, 0.13), (0.21, 0.07)):
        k = int(d * SR)
        out[k:] += out[:-k].copy() * g
    out /= np.abs(out).max() / 0.9
    w = wave.open(path, "wb")
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes()); w.close()


# ======================= CLI =======================
def score(st, leads, counts):
    if st["final"] is None or st["t_over"] is None:
        return -1e9, "sem explosão"
    te = st["t_expl"]
    fa, fd = st["final"]
    margin = abs(fa - fd) / max(1, fa + fd)
    late = [l for l in leads if l[0] > 22]
    sc = 0
    sc -= abs(st["t_over"] - 34) * 1.2          # lotar perto dos 34 s
    sc += min(len(leads), 7) * 1.0
    sc += min(len(late), 4) * 2.0
    sc -= margin * 25
    if te > 46:
        sc -= (te - 46) * 3
    return sc, f"cheio {st['t_over']:.1f} explode {te:.1f} viradas {[round(l[0], 1) for l in leads]} final {st['final']}"


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    ar = Arena(CONFIGS[name])
    mult = int(os.environ.get("MULT", "3"))
    od = f"out_{name}"
    os.makedirs(od, exist_ok=True)
    if mode == "info":
        print(name, "C", ar.C, "r_in", round(ar.r_in), "RB", ar.RB, "CAP", ar.CAP, "R1", round(ar.R1), "R2", round(ar.R2), "D", round(ar.D), "Wb", round(ar.Wb))
    elif mode == "search":
        for seed in range(int(sys.argv[3]), int(sys.argv[4])):
            _, au, pops, leads, st, counts = simulate(ar, seed, mult, keep=False)
            sc, desc = score(st, leads, counts)
            print(name, "MULT", mult, "seed", seed, "score", round(sc, 1), desc, flush=True)
    elif mode in ("render", "preview"):
        seed = int(sys.argv[3])
        frames, audio, pops, leads, st, counts = simulate(ar, seed, mult)
        print(name, "duracao", len(frames) / FPS, score(st, leads, counts)[1], flush=True)
        rd = Renderer(ar)
        srng = np.random.default_rng(1)
        conf = Conf(3, (st["t_expl"] or 0) + 0.8, ar.cfg["col"])
        if mode == "preview":
            want = set(int(x) for x in sys.argv[4:])
            for i, fr in enumerate(frames):
                if i in want or i == len(frames) - 1:
                    im = rd.render(fr, pops, leads, st, srng, conf)
                    cv2.imwrite(f"{od}/p_{i:04d}.png", im)
                else:
                    na, nd = fr["counts"]; rd.disp += (na / max(1, na + nd) - rd.disp) * 0.15
        else:
            enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                    "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                    "-pix_fmt", "yuv420p", f"{od}/v_tmp.mp4"], stdin=subprocess.PIPE)
            for fr in frames:
                enc.stdin.write(rd.render(fr, pops, leads, st, srng, conf).tobytes())
            enc.stdin.close(); enc.wait()
            dur = len(frames) / FPS
            make_audio(sorted(audio), dur, ar.cfg["snd"], f"{od}/audio.wav")
            final = f"{od}/duelo_{name}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{od}/v_tmp.mp4", "-i", f"{od}/audio.wav", "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(dur), f"{od}/tmp.mp4"], check=True)
            os.replace(f"{od}/tmp.mp4", final)
            print("ok", final, flush=True)
