"""DUELO 😇 x 😈 num hexágono.
Rotor de 3 braços (passa por cima) com bolsos em U e martelos nas pontas. A cada 1,5 s os três
bolsos prensam ao mesmo tempo: no bolso, quem estiver em maioria leva os outros junto e cada
um vira MULT do time vencedor (viradas!). Placar ao vivo, 'VIROU!', lotou -> 3 prensadas
tremendo -> explode -> tela do vencedor."""
import sys, os, math, json, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

W, H, FPS = 720, 1280, 30
DUR = 52.0
NF = int(DUR * FPS)
C = np.array([360.0, 650.0])
A0 = 300.0                       # apótema do hexágono (centro -> lado)
NRM = np.array([[math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)])
RB = 12.0
SPR_D = 27
GEAR_R = 40.0
ROD_TH = 15.0
PK_TH = 13.0
PK_L = 92.0
PL = 60.0
R_P0 = math.sqrt((A0 - 3 - PK_TH / 2) ** 2 - PK_L ** 2) - PL
HM_TH = 15.0
PUNCH = 46.0
PHI_P = 0.30
PERIOD = 9.0                     # 1 volta / 9 s -> prensada (3 bolsos) a cada 1,5 s
OMEGA = 2 * math.pi / PERIOD
T_PRESS0 = 1.5
T_TRAP = PHI_P / OMEGA
SUB = 3
DT = 1.0 / (FPS * SUB)
MULT = int(os.environ.get("MULT", "2"))
SWIRL = float(os.environ.get("SWIRL", "0.4"))
OUTF = float(os.environ.get("OUTF", "110"))
CAP = int(os.environ.get("CAP", "640"))
A_LEVELS = [A0 + 4, A0 + 12, A0 + 20, A0 + 28]
TEAM_COL = [(255, 200, 80), (210, 50, 175)]      # BGR: anjo azul-claro, diabo roxo


def theta_at(t):
    return OMEGA * (t - T_PRESS0)


def phi_at(t):
    return (theta_at(t) + math.pi / 6) % (math.pi / 3) - math.pi / 6


PRESS_T = [T_PRESS0 + k * PERIOD / 6 for k in range(int(DUR / (PERIOD / 6)) + 2)]


def hammer_r(t):
    b = max(0.0, 1 - abs(phi_at(t)) / PHI_P)
    return R_P0 + 10 + PUNCH * b * b * (3 - 2 * b)


def mech_at(t):
    th = theta_at(t)
    rh = hammer_r(t)
    ends = []
    for k in range(3):
        a = th + k * 2 * math.pi / 3
        ue = np.array([math.cos(a), math.sin(a)]); te = np.array([-ue[1], ue[0]])
        B = C + ue * R_P0
        segs = [(B - te * PK_L, B + te * PK_L, PK_TH),
                (B - te * PK_L, B - te * PK_L + ue * PL, PK_TH),
                (B + te * PK_L, B + te * PK_L + ue * PL, PK_TH),
                (C + ue * rh - te * (PK_L - 11), C + ue * rh + te * (PK_L - 11), HM_TH)]
        ends.append(dict(k=k, ue=ue, te=te, B=B, rh=rh, hm=C + ue * rh, segs=segs))
    return th, ends


def push_capsule(pos, mask, a, b, th):
    ab = b - a
    s = np.clip(((pos - a) @ ab) / (ab @ ab), 0, 1)
    cp = a + s[:, None] * ab
    d = pos - cp
    dist = np.linalg.norm(d, axis=1)
    lim = RB + th / 2
    hit = mask & (dist < lim)
    if hit.any():
        pos[hit] = cp[hit] + d[hit] / np.maximum(dist[hit], 1e-4)[:, None] * lim


# ======================= simulação =======================
def simulate(seed, keep=True):
    rng = np.random.default_rng(seed)
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2)); ang = np.zeros(0)
    team = np.zeros(0, np.int8); ghost = np.zeros(0, bool); trap = np.zeros(0, np.int8)
    frames, audio, pops, leads = [], [], [], []
    st = dict(A=A0, exploded=False, t_over=None, t_expl=None, lvl=0, t_lastpop=None, leader=None, winner=None,
              final=None)
    frags, spawn_q, done = [], [], set()
    counts = []

    def add(n, p0, v0, tm):
        nonlocal pos, prev, ang, team, ghost, trap
        p = p0 + rng.normal(0, 5, (n, 2))
        v = v0 + rng.normal(0, 70, (n, 2))
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        ang = np.concatenate([ang, rng.uniform(-0.5, 0.5, n)])
        team = np.concatenate([team, np.full(n, tm, np.int8)])
        ghost = np.concatenate([ghost, np.zeros(n, bool)])
        trap = np.concatenate([trap, np.zeros(n, np.int8)])

    # um de cada time, cada um num bolso da primeira prensada
    th0, ends0 = mech_at(PRESS_T[0])
    for tm, k in ((0, 0), (1, 1)):
        e = ends0[k]
        add(1, C + e["ue"] * (A0 - RB - 12) + e["te"] * 6, np.zeros(2), tm)
    pos = pos - 0  # sem ruído inicial
    pos[:] = [C + ends0[k]["ue"] * (A0 - RB - 12) for k in (0, 1)]
    prev = pos.copy(); ghost[:] = True
    t_tick, tick_i = 0.5, 0

    for fi in range(NF):
        t = fi / FPS
        ex = st["exploded"]
        while t_tick <= t:
            if not ex:
                audio.append((t_tick, "tick" if tick_i % 2 == 0 else "tock", 0.12)); tick_i += 1
            if st["t_over"] is not None and not ex:
                t_tick += 0.07 if st["lvl"] >= 3 else 0.5 / (1 + st["lvl"] * 1.3)
            else:
                t_tick += 0.5
        th, ends = mech_at(t)

        if not ex:
            for k, tp in enumerate(PRESS_T):
                if ("trap", k) not in done and t >= tp - T_TRAP:
                    done.add(("trap", k)); ghost[:] = False
                    audio.append((tp - T_TRAP, "clack", 0.3))
                    for en in ends:
                        d = pos - C
                        a = d @ en["te"]; q = d @ en["ue"]
                        ins = (np.abs(a) < PK_L - PK_TH / 2) & (q > en["rh"] + HM_TH / 2) & (trap == 0)
                        trap[ins] = en["k"] + 1
                if ("press", k) not in done and t >= tp:
                    done.add(("press", k))
                    audio.append((tp, "press", 0.4))
                    if not (trap != 0).any():
                        continue
                    grow = st["t_over"] is not None
                    if grow:
                        st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                        audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                    for en in ends:
                        sel = trap == en["k"] + 1
                        nt = int(sel.sum())
                        if not nt:
                            continue
                        na = int((sel & (team == 0)).sum()); nd = nt - na
                        pos_sel = pos[sel]; team_sel = team[sel]
                        km = ~sel
                        pos, prev, ang, team, ghost, trap = pos[km], prev[km], ang[km], team[km], ghost[km], trap[km]
                        # quem está encostado no martelo leva o bolso inteiro
                        qsel = (pos_sel - C) @ en["ue"]
                        w = int(team_sel[int(np.argmin(qsel))])
                        groups = [(w, nt * MULT)]
                        if min(na, nd) > 0:
                            audio.append((t + 0.02, "steal", 0.6))
                        for tm, n in groups:
                            if grow:
                                capn = int(0.84 * 2 * math.sqrt(3) * A_LEVELS[st["lvl"]] ** 2 / (math.pi * RB * RB))
                                n = max(1, min(n, (capn - len(pos)) // 3 + 2))
                            else:
                                n = max(1, min(n, CAP + 20 - len(pos)))
                            pulses = max(1, min(8, (n + 2) // 3))
                            base, rest = divmod(n, pulses)
                            for p in range(pulses):
                                cnt = base + (1 if p < rest else 0)
                                side = 1 if p % 2 == 0 else -1
                                p0 = C + en["ue"] * (R_P0 + PL - 6) + en["te"] * side * (PK_L + RB + 12)
                                v0 = en["te"] * side * 260 - en["ue"] * 130
                                spawn_q.append([t + p * 0.07, cnt, p0, v0, tm])
                            pops.append((t, (en["hm"] + en["ue"] * 28).tolist(), n, tm, min(na, nd) > 0))
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, p0, v0, tm = spawn_q.pop(0)
            if not st["exploded"] and n > 0:
                add(n, p0, v0, tm)
                audio.append((ts, "angel" if tm == 0 else "devil", 0.5))

        na = int((team == 0).sum()); nd = int((team == 1).sum())
        if not ex:
            if na != nd and (na + nd) >= 12:
                ld = 0 if na > nd else 1
                if st["leader"] is not None and ld != st["leader"]:
                    leads.append((t, ld)); audio.append((t, "lead", 0.55))
                st["leader"] = ld
            elif st["leader"] is None and na != nd:
                st["leader"] = 0 if na > nd else 1

        if not ex and st["t_over"] is None and len(pos) >= CAP:
            st["t_over"] = t; audio.append((t, "rumble", 0.5))
        if not ex and st["t_over"] is not None:
            st["A"] += (A_LEVELS[st["lvl"]] - st["A"]) * 0.06
            if st["lvl"] >= 3 and t >= st["t_lastpop"] + 1.0:
                st.update(exploded=True, t_expl=t); ex = True
                st["final"] = (na, nd)
                st["winner"] = 0 if na >= nd else 1
                audio.append((t, "boom", 1.0)); audio.append((t + 0.9, "fanfare", 0.9))
                d = pos - C
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                spd = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / st["A"])
                prev = pos - (d / dist * spd + rng.normal(0, 120, pos.shape)) * DT
                vr = st["A"] / math.cos(math.pi / 6)
                for k in range(6):
                    a0 = math.radians(60 * k - 30); a1 = math.radians(60 * k + 30)
                    P0 = C + vr * np.array([math.cos(a0), math.sin(a0)]); P1 = C + vr * np.array([math.cos(a1), math.sin(a1)])
                    for q in range(4):
                        a = P0 + (P1 - P0) * q / 4; b = P0 + (P1 - P0) * (q + 0.9) / 4
                        m = (a + b) / 2; dv = (m - C) / np.linalg.norm(m - C)
                        frags.append(dict(a=a - m, b=b - m, c=m, rot=0.0, v=dv * rng.uniform(500, 1100), w=rng.uniform(-6, 6)))

        grav = np.array([0.0, 900.0]) if ex else np.zeros(2)
        for s in range(SUB):
            if len(pos) == 0:
                break
            ts = t + s * DT
            v = (pos - prev) / DT * 0.9993
            if not ex:
                rel = pos - C
                vt = np.stack([-rel[:, 1], rel[:, 0]], 1) * SWIRL
                v += (vt - v) * min(1.0, 0.8 * DT)
                v += rel / A0 * OUTF * DT
                v[ghost] = 0
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            cap = 2500 if ex else 700
            v = np.where(spd > cap, v / np.maximum(spd, 1e-6) * cap, v)
            prev = pos; pos = pos + v * DT + grav * DT * DT
            ang = ang + v[:, 0] * DT / RB * 0.35
            _, ends_s = mech_at(ts)
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
                    for en in ends_s:
                        for a, b, thk in en["segs"]:
                            push_capsule(pos, free, a, b, thk)
                        sel = trap == en["k"] + 1
                        if sel.any():
                            d = pos[sel] - C
                            a = np.clip(d @ en["te"], -(PK_L - PK_TH / 2 - RB), PK_L - PK_TH / 2 - RB)
                            q = np.maximum(d @ en["ue"], en["rh"] + HM_TH / 2 + RB * 0.6)
                            pos[sel] = C + a[:, None] * en["te"] + q[:, None] * en["ue"]
                    dg = pos - C
                    dd = np.linalg.norm(dg, axis=1)
                    hg = solid & (dd < RB + GEAR_R)
                    if hg.any():
                        pos[hg] = C + dg[hg] / np.maximum(dd[hg], 1e-4)[:, None] * (RB + GEAR_R)
                    lim = st["A"] - RB
                    for nrm in NRM:
                        q = (pos - C) @ nrm
                        o = q > lim
                        if o.any():
                            pos[o] -= ((q[o] - lim)[:, None]) * nrm
                            vv = pos[o] - prev[o]
                            vn = vv @ nrm
                            vv = vv - np.maximum(vn, 0)[:, None] * nrm * 1.85
                            prev[o] = pos[o] - vv
        if ex:
            kp = (pos[:, 0] > -60) & (pos[:, 0] < W + 60) & (pos[:, 1] < H + 60) & (pos[:, 1] > -600)
            pos, prev, ang, team, ghost, trap = pos[kp], prev[kp], ang[kp], team[kp], ghost[kp], trap[kp]
            for fr in frags:
                fr["c"] = fr["c"] + fr["v"] / FPS; fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS
                fr["rot"] += fr["w"] / FPS
        counts.append((na, nd) if not ex else st["final"])
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), ang=ang.copy(), team=team.copy(), A=st["A"], exploded=ex,
                               t_over=st["t_over"], lvl=st["lvl"], t_lastpop=st["t_lastpop"], t_expl=st["t_expl"],
                               frags=[dict(f) for f in frags], counts=counts[-1]))
        if ex and t > st["t_expl"] + 4.6:
            break
    return frames, audio, pops, leads, st, counts


# ======================= desenho =======================
EMOJI_FONT = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)
FONT_B = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"


def emoji_img(ch):
    im = Image.new("RGBA", (170, 170), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=EMOJI_FONT, embedded_color=True)
    im = im.crop(im.getbbox())
    s = max(im.size)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0)); sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return sq


def to_pre(im):
    a = np.asarray(im).astype(np.float32)
    al = a[..., 3:4] / 255
    return (a[..., [2, 1, 0]] * al).astype(np.float32), (1 - al).astype(np.float32)


NROT = 24
EMO = [emoji_img("\U0001F607"), emoji_img("\U0001F608")]
SPR = []
for im in EMO:
    big = im.resize((SPR_D * 4, SPR_D * 4), Image.LANCZOS)
    SPR.append([to_pre(big.rotate(-360 * r / NROT, resample=Image.BICUBIC).resize((SPR_D, SPR_D), Image.LANCZOS)) for r in range(NROT)])
HALF = SPR_D // 2
CROWN = emoji_img("\U0001F451")
FINGER = emoji_img("\U0001F447")
SH = 16
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


def emoji_scaled(i_img, d):
    key = ("e", id(i_img), d)
    if key not in _cache:
        _cache[key] = to_pre(i_img.resize((d, d), Image.LANCZOS))
    return _cache[key]


def text_img(txt, size, color=(255, 255, 255), stroke=0, stroke_col=(0, 0, 0)):
    key = ("t", txt, size, color, stroke, stroke_col)
    if key not in _cache:
        font = ImageFont.truetype(FONT_B, size)
        bb = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), txt, font=font, stroke_width=stroke)
        im = Image.new("RGBA", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), txt, font=font, fill=color[::-1] + (255,) if False else (color[2], color[1], color[0], 255),
                                stroke_width=stroke, stroke_fill=(stroke_col[2], stroke_col[1], stroke_col[0], 255))
        _cache[key] = to_pre(im)
    return _cache[key]


def put_text(f, txt, size, cx, cy, color=(255, 255, 255), stroke=0, alpha=1.0, stroke_col=(0, 0, 0), scale=1.0):
    pre, inv = text_img(txt, size, color, stroke, stroke_col)
    if abs(scale - 1) > 0.02:
        h, w = pre.shape[:2]
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pre = cv2.resize(pre, (nw, nh)); inv = cv2.resize(inv[..., 0], (nw, nh))[..., None]
    h, w = pre.shape[:2]
    blit(f, pre, inv, int(cx - w / 2), int(cy - h / 2), alpha)


def make_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - C[0]) ** 2 + (yy - C[1]) ** 2) / 700
    g = np.clip(1 - d, 0, 1) ** 1.5
    bg = np.zeros((H, W, 3), np.float32)
    bg[:] = (14, 6, 10)
    bg += g[..., None] * np.array([48, 18, 34], np.float32)
    return bg


BG = make_bg()


def make_glow(r):
    s = int(r * 2)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    return np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 2


GLOW = make_glow(140)


def add_glow(f, center, color, k, g=GLOW):
    s = g.shape[0]
    x0, y0 = int(center[0] - s / 2), int(center[1] - s / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if xa >= xb or ya >= yb or k <= 0.01:
        return
    f[ya:yb, xa:xb] += g[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * k * np.array(color, np.float32)


def gear_poly(center, a0, r, teeth):
    pts = []
    for i in range(teeth * 4):
        a = a0 + i * (2 * math.pi / (teeth * 4))
        rr = r if (i % 4) in (0, 1) else r * 0.82
        pts.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a)))
    return (np.array(pts) * SH).astype(np.int32)


def draw_rod(img, a, b, th, base=(170, 165, 170)):
    d = b - a; L = float(np.linalg.norm(d)); u = d / L; n = np.array([-u[1], u[0]]); h = th / 2
    cv2.line(img, P(a), P(b), (60, 55, 60), int(th + 3), cv2.LINE_AA, 4)
    cv2.line(img, P(a), P(b), base, int(th), cv2.LINE_AA, 4)
    cv2.line(img, P(a + n * (h - 3)), P(b + n * (h - 3)), tuple(min(255, c + 50) for c in base), 2, cv2.LINE_AA, 4)
    cv2.line(img, P(a - n * (h - 3)), P(b - n * (h - 3)), tuple(int(c * 0.65) for c in base), 2, cv2.LINE_AA, 4)


def hex_pts(a, wob, t):
    vr = a / math.cos(math.pi / 6)
    pts = []
    for k in range(6):
        a0 = math.radians(60 * k - 30); a1 = math.radians(60 * k + 30)
        P0 = np.array([math.cos(a0), math.sin(a0)]) * vr; P1 = np.array([math.cos(a1), math.sin(a1)]) * vr
        nrm = NRM[k]
        for s in np.linspace(0, 1, 30, endpoint=False):
            bump = math.sin(math.pi * s) * wob * (0.7 + 0.3 * math.sin(t * 25 + k * 1.7 + s * 9))
            pts.append(C + P0 + (P1 - P0) * s + nrm * bump)
    return np.array(pts)


def ease_out_back(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Conf:
    def __init__(self, seed, t0):
        r = np.random.default_rng(seed)
        n = 160
        self.t0 = t0
        self.p0 = np.stack([r.uniform(0, W, n), r.uniform(-500, -20, n)], 1)
        self.v = np.stack([r.normal(0, 60, n), r.uniform(260, 520, n)], 1)
        self.col = [tuple(int(c) for c in r.choice([[60, 60, 255], [60, 220, 255], [255, 200, 80], [210, 50, 175], [90, 230, 90], [255, 255, 255]]))
                    for _ in range(n)]
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
                hw, hh = 7 * abs(math.cos(a)) + 1, 4
                box = cv2.boxPoints(((x, y), (hw * 2, hh * 2), math.degrees(a * 0.7)))
                cv2.fillPoly(img, [box.astype(np.int32)], self.col[i], cv2.LINE_AA)


def render(fr, pops, leads, st, srng, conf, disp):
    t = fr["t"]; ex = fr["exploded"]
    f = BG.copy()
    for (t0, gc, n, tm, steal) in pops:
        if t0 - 0.02 <= t < t0 + 0.9:
            k = 1.0 if t < t0 + 0.12 else max(0.0, 1 - (t - t0 - 0.12) / 0.78)
            add_glow(f, gc, TEAM_COL[tm], k * (1.1 if steal else 0.7))
    ip = np.round(fr["pos"]).astype(np.int32)
    ri = (np.round(fr["ang"] / (2 * math.pi) * NROT).astype(int)) % NROT
    tm_ = fr["team"]
    for k in range(len(ip)):
        pre, inv = SPR[tm_[k]][ri[k]]
        x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
        if 0 <= x0 and 0 <= y0 and x0 + SPR_D <= W and y0 + SPR_D <= H:
            reg = f[y0:y0 + SPR_D, x0:x0 + SPR_D]; reg *= inv; reg += pre
        else:
            blit(f, pre, inv, x0, y0)
    img = np.clip(f, 0, 255).astype(np.uint8)

    na, nd = fr["counts"]
    tot = max(1, na + nd)
    share = na / tot
    disp["share"] += (share - disp["share"]) * 0.15
    lead_col = TEAM_COL[0] if na >= nd else TEAM_COL[1]

    wob = 0.0
    if fr["t_over"] is not None and not ex:
        lv = fr["lvl"]
        kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.6) if fr["t_lastpop"] is not None else 0.0
        kick0 = max(0.0, 1 - (t - fr["t_over"]) / 0.8)
        wob = [3, 6, 10, 15][lv] + 10 * kick + 8 * kick0
        if lv >= 3:
            wob += 10 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
    if not ex:
        pts = (hex_pts(fr["A"], wob, t) * SH).astype(np.int32)
        glow = np.zeros_like(img)
        cv2.polylines(glow, [pts], True, lead_col, 14, cv2.LINE_AA, 4)
        glow = cv2.GaussianBlur(glow, (0, 0), 9)
        img = cv2.add(img, (glow * 0.8).astype(np.uint8))
        cv2.polylines(img, [pts], True, (245, 240, 250), 4, cv2.LINE_AA, 4)
    for fg in fr["frags"]:
        cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"])
        R = np.array([[cr, -sr], [sr, cr]])
        cv2.line(img, P(fg["c"] + R @ fg["a"]), P(fg["c"] + R @ fg["b"]), (245, 240, 250), 4, cv2.LINE_AA, 4)

    ta = 1.0 if not ex else max(0.0, 1 - (t - fr["t_expl"]) / 0.25)
    if ta > 0:
        th, ends = mech_at(t)
        ov = img.copy()
        for en in ends:
            cv2.line(ov, P(C + np.array([5.0, 7.0])), P(en["B"] + np.array([5.0, 7.0])), (8, 4, 6), int(ROD_TH + 4), cv2.LINE_AA, 4)
        for en in ends:
            ue, te, B = en["ue"], en["te"], en["B"]
            draw_rod(ov, B, en["hm"], 8, base=(120, 115, 120))
            draw_rod(ov, en["hm"] - te * (PK_L - 11), en["hm"] + te * (PK_L - 11), HM_TH, base=(70, 180, 235))
            c1, c2 = B - te * PK_L, B + te * PK_L
            draw_rod(ov, c1, c2, PK_TH); draw_rod(ov, c1, c1 + ue * PL, PK_TH); draw_rod(ov, c2, c2 + ue * PL, PK_TH)
            for cc in (c1, c2):
                cv2.circle(ov, P(cc), int(PK_TH / 2 * SH), (170, 165, 170), -1, cv2.LINE_AA, 4)
                cv2.circle(ov, P(cc), int(3 * SH), (70, 65, 70), -1, cv2.LINE_AA, 4)
            draw_rod(ov, C, B, ROD_TH)
        gp = gear_poly(C, th, GEAR_R, 15)
        cv2.fillPoly(ov, [gp], (70, 175, 215), cv2.LINE_AA, 4)
        cv2.polylines(ov, [gp], True, (30, 90, 120), 1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(C), int(GEAR_R * 0.6 * SH), (40, 120, 160), 2, cv2.LINE_AA, 4)
        for q in range(3):
            a = th + q * 2 * math.pi / 3 + math.pi / 3
            cv2.circle(ov, P(C + 0.4 * GEAR_R * np.array([math.cos(a), math.sin(a)])), int(5 * SH), (30, 100, 140), -1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(C), int(12 * SH), (200, 195, 200), -1, cv2.LINE_AA, 4)
        img = ov if ta >= 1 else cv2.addWeighted(ov, ta, img, 1 - ta, 0)

    f = img.astype(np.float32)
    # "+N" de cada prensada
    for (t0, gc, n, tm, steal) in pops:
        dtt = t - t0
        if 0 <= dtt < 0.9:
            gc = np.array(gc)
            p = gc - (gc - C) / max(1e-3, np.linalg.norm(gc - C)) * (45 + dtt * 60)
            al = min(1.0, (1 - dtt / 0.9) * 1.6)
            put_text(f, f"+{n}", 30 if not steal else 36, p[0], p[1], TEAM_COL[tm], stroke=4, alpha=al, stroke_col=(20, 10, 15))

    # ---------- placar ----------
    put_text(f, "BEM  vs  MAL", 46, W / 2, 62, (255, 255, 255), stroke=3, stroke_col=(40, 20, 30))
    bx0, bx1, by = 70, 650, 214
    cv2_f = f
    # ícones e números
    ia = emoji_scaled(EMO[0], 64); idv = emoji_scaled(EMO[1], 64)
    bump_a = bump_d = 1.0
    for (t0, gc, n, tm, steal) in pops:
        if 0 <= t - t0 < 0.25:
            if tm == 0: bump_a = 1.18
            else: bump_d = 1.18
    blit(f, *ia, 40, 112); blit(f, *idv, W - 104, 112)
    put_text(f, str(na), 52, 205, 146, TEAM_COL[0], stroke=4, stroke_col=(30, 15, 20), scale=bump_a)
    put_text(f, str(nd), 52, W - 205, 146, TEAM_COL[1], stroke=4, stroke_col=(30, 15, 20), scale=bump_d)
    # barra cabo-de-guerra
    img = np.clip(f, 0, 255).astype(np.uint8)
    xs = int(bx0 + (bx1 - bx0) * disp["share"])
    cv2.rectangle(img, (bx0 - 3, by - 17), (bx1 + 3, by + 17), (40, 25, 35), -1, cv2.LINE_AA)
    cv2.rectangle(img, (bx0, by - 14), (xs, by + 14), TEAM_COL[0], -1, cv2.LINE_AA)
    cv2.rectangle(img, (xs, by - 14), (bx1, by + 14), TEAM_COL[1], -1, cv2.LINE_AA)
    hl = img[by - 13:by - 5, bx0:bx1].astype(np.float32)
    img[by - 13:by - 5, bx0:bx1] = np.clip(hl + 45, 0, 255).astype(np.uint8)
    cv2.circle(img, (xs, by), 19, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (xs, by), 19, (40, 25, 35), 2, cv2.LINE_AA)
    f = img.astype(np.float32)
    put_text(f, f"{round(100 * na / tot)}%", 20, bx0 + 40, by, (255, 255, 255), stroke=2, stroke_col=(60, 40, 20))
    put_text(f, f"{round(100 * nd / tot)}%", 20, bx1 - 40, by, (255, 255, 255), stroke=2, stroke_col=(60, 20, 50))
    put_text(f, "VS", 18, xs, by, (40, 25, 35))

    # "VIROU!"
    recent = [(t0, ld) for (t0, ld) in leads if 0 <= t - t0 < 1.1]
    for (t0, ld) in recent[-1:]:
        dtt = t - t0
        if not ex:
            sc = ease_out_back(min(1.0, dtt / 0.25))
            al = 1.0 if dtt < 0.8 else max(0.0, 1 - (dtt - 0.8) / 0.3)
            put_text(f, "VIROU!", 64, W / 2, 1060, TEAM_COL[ld], stroke=6, stroke_col=(255, 255, 255), alpha=al, scale=max(0.05, sc))

    # rodapé
    if not ex:
        recent_flip = any(0 <= t - t0 < 1.1 for (t0, ld) in leads)
        if not recent_flip:
            if fr["t_over"] is not None:
                pulse = 1 + 0.08 * math.sin(t * 18)
                put_text(f, "VAI EXPLODIR!", 50, W / 2, 1070, (60, 60, 255), stroke=5, stroke_col=(255, 255, 255), scale=pulse)
            else:
                pulse = 1 + 0.04 * math.sin(t * 5)
                put_text(f, "Quem vence?", 44, W / 2 - 30, 1060, (255, 255, 255), stroke=3, stroke_col=(40, 20, 30), scale=pulse)
                pre, inv = emoji_scaled(FINGER, 52)
                blit(f, pre, inv, int(W / 2 + 130), int(1060 - 26 + 6 * math.sin(t * 6)))
                put_text(f, "comenta seu palpite", 26, W / 2, 1120, (200, 190, 210))

    img = np.clip(f, 0, 255).astype(np.uint8)

    # explosão + tela do vencedor
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
                pre, inv = emoji_scaled(EMO[wv], max(4, d))
                add_glow(f, (W / 2, 640), TEAM_COL[wv], 1.4 * min(1, sc), make_glow_cached(220))
                blit(f, pre, inv, int(W / 2 - d / 2), int(650 - d / 2))
                if te > 1.2:
                    cs = ease_out_back(min(1.0, (te - 1.2) / 0.35))
                    cd = int(150 * max(0.05, cs))
                    pre, inv = emoji_scaled(CROWN, cd)
                    blit(f, pre, inv, int(W / 2 - cd / 2), int(650 - d / 2 - cd * 0.75 + 10 * math.sin(te * 4)))
                put_text(f, "VENCEU!", 92, W / 2, 920, TEAM_COL[wv], stroke=7, stroke_col=(255, 255, 255), scale=max(0.05, sc))
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


_GC = {}


def make_glow_cached(r):
    if r not in _GC:
        _GC[r] = make_glow(r)
    return _GC[r]


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "search":
        res = []
        for seed in range(int(sys.argv[2]), int(sys.argv[3])):
            _, au, pops, leads, st, counts = simulate(seed, keep=False)
            full = [a[0] for a in au if a[1] == "rumble"]
            late = [round(l[0], 1) for l in leads]
            fin = st["final"]
            print(seed, "cheio", round(full[0], 1) if full else None, "explode", st["t_expl"] and round(st["t_expl"], 1),
                  "viradas", late, "final", fin, flush=True)
        sys.exit()
    seed = int(sys.argv[2])
    frames, audio, pops, leads, st, counts = simulate(seed)
    print("duracao", len(frames) / FPS, "explosao", st["t_expl"], "final", st["final"], "viradas", [round(l[0], 1) for l in leads], flush=True)
    srng = np.random.default_rng(1)
    conf = Conf(3, (st["t_expl"] or 0) + 0.8)
    disp = dict(share=0.5)
    if mode == "preview":
        want = set(int(x) for x in sys.argv[3:])
        for i, fr in enumerate(frames):
            im = render(fr, pops, leads, st, srng, conf, disp)
            if i in want:
                cv2.imwrite(f"out/p_{i:04d}.png", im)
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "out/v_tmp.mp4"], stdin=subprocess.PIPE)
        for fr in frames:
            enc.stdin.write(render(fr, pops, leads, st, srng, conf, disp).tobytes())
        enc.stdin.close(); enc.wait()
        json.dump(dict(dur=len(frames) / FPS, events=sorted(audio)), open("out/audio_events.json", "w"))
        print("ok", enc.returncode)
