"""🤑 que se multiplicam num quadrado.
Engrenagem central gira um bastão sempre no mesmo sentido; na ponta, um martelo dobrado (T).
A cada passagem por um lado do quadrado o martelo fica rente à parede e esmaga os 🤑 que
estão ali; cada um vira MULT. Lotou -> borda balança -> mais 3 prensadas -> explode (ka-ching)."""
import sys, os, math, json, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

W, H, FPS = 720, 1280, 30
DUR = 56.0
NF = int(DUR * FPS)
C = np.array([360.0, 640.0])
HS0 = 310.0                    # meio lado do quadrado
RB = 12.0                      # raio de colisão do emoji
SPR_D = 27                     # diâmetro do desenho
GEAR_R = 42.0
ROD_TH = 16.0
PK_TH = 14.0                   # espessura do suporte em "U"
PK_L = 100.0                   # meia largura do bolso
PL = 62.0                      # comprimento das pontas dobradas
R_P0 = math.sqrt((HS0 - 3 - PK_TH / 2) ** 2 - PK_L ** 2) - PL   # fundo do "U"
HM_TH = 16.0
PUNCH = 46.0                   # quanto o martelo avança na prensada
PHI_P = 0.30
PERIOD = 6.0                   # uma volta a cada 6 s; as duas pontas prensam a cada 1,5 s
OMEGA = 2 * math.pi / PERIOD
T_PRESS0 = 1.5
T_TRAP = PHI_P / OMEGA         # o bolso fecha um pouco antes da prensada
SUB = 3
DT = 1.0 / (FPS * SUB)
MULT = int(os.environ.get("MULT", "4"))
SWIRL = float(os.environ.get("SWIRL", "0.45"))
OUTF = float(os.environ.get("OUTF", "120"))
CAP = int(os.environ.get("CAP", "760"))
HS_LEVELS = [HS0 + 4, HS0 + 12, HS0 + 20, HS0 + 28]
GOLD = (60, 200, 255)          # BGR


def theta_at(t):
    return OMEGA * (t - T_PRESS0)            # gira sempre no sentido horário


PRESS_T = [T_PRESS0 + k * PERIOD / 4 for k in range(int(DUR / (PERIOD / 4)) + 2)]


def phi_at(t):
    return (theta_at(t) + math.pi / 4) % (math.pi / 2) - math.pi / 4


def hammer_r(t):
    ph = phi_at(t)
    b = max(0.0, 1 - abs(ph) / PHI_P)
    return R_P0 + 10 + PUNCH * b * b * (3 - 2 * b)


def mech_at(t):
    th = theta_at(t)
    u = np.array([math.cos(th), math.sin(th)])
    tv = np.array([-u[1], u[0]])
    rh = hammer_r(t)
    ends = []
    for e in (1, -1):
        ue, te = u * e, tv * e
        B = C + ue * R_P0
        segs = [(B - te * PK_L, B + te * PK_L, PK_TH),
                (B - te * PK_L, B - te * PK_L + ue * PL, PK_TH),
                (B + te * PK_L, B + te * PK_L + ue * PL, PK_TH),
                (C + ue * rh - te * (PK_L - 12), C + ue * rh + te * (PK_L - 12), HM_TH)]
        ends.append(dict(e=e, ue=ue, te=te, B=B, rh=rh, hm=C + ue * rh, segs=segs))
    return th, u, tv, ends


def push_capsule(pos, mask, a, b, th):
    ab = b - a
    L2 = ab @ ab
    s = np.clip(((pos - a) @ ab) / L2, 0, 1)
    cp = a + s[:, None] * ab
    d = pos - cp
    dist = np.linalg.norm(d, axis=1)
    lim = RB + th / 2
    hit = mask & (dist < lim)
    if hit.any():
        n = d[hit] / np.maximum(dist[hit], 1e-4)[:, None]
        pos[hit] = cp[hit] + n * lim


# ---------------- simulação ----------------
def simulate(seed, keep=True):
    rng = np.random.default_rng(seed)
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2)); ang = np.zeros(0)
    ghost = np.zeros(0, bool); trap = np.zeros(0, np.int8)
    frames, audio, pops = [], [], []
    st = dict(HS=HS0, exploded=False, t_over=None, t_expl=None, t_restart=None, lvl=0,
              t_lastpop=None, restart0=None)
    frags, bills, spawn_q = [], [], []
    done = set()
    coin_i = [0]

    def add(n, p0, v0):
        nonlocal pos, prev, ang, ghost, trap
        p = p0 + rng.normal(0, 5, (n, 2))
        v = v0 + rng.normal(0, 70, (n, 2))
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        ang = np.concatenate([ang, rng.uniform(-0.5, 0.5, n)])
        ghost = np.concatenate([ghost, np.zeros(n, bool)])
        trap = np.concatenate([trap, np.zeros(n, np.int8)])

    def place_single(tp):
        nonlocal pos, prev, ang, ghost, trap
        th, u, tv, ends = mech_at(tp)
        p = C + u * (HS0 - RB - 12) + tv * 8
        pos = p[None].copy(); prev = pos.copy(); ang = np.zeros(1); ghost = np.ones(1, bool); trap = np.zeros(1, np.int8)

    place_single(PRESS_T[0])
    t_tick, tick_i = 0.5, 0

    for fi in range(NF):
        t = fi / FPS
        ex = st["exploded"]
        while t_tick <= t:
            if not (st["t_expl"] and st["t_expl"] <= t_tick < (st["t_restart"] or 99)):
                audio.append((t_tick, "tick" if tick_i % 2 == 0 else "tock", 0.16)); tick_i += 1
            if st["t_over"] is not None and not ex:
                t_tick += 0.07 if st["lvl"] >= 3 else 0.5 / (1 + st["lvl"] * 1.3)
            else:
                t_tick += 0.5
        th, u, tv, ends = mech_at(t)

        if not ex:
            for k, tp in enumerate(PRESS_T):
                if ("trap", k) not in done and t >= tp - T_TRAP:
                    done.add(("trap", k))
                    ghost[:] = False
                    audio.append((tp - T_TRAP, "clack", 0.35))
                    for en in ends:
                        d = pos - C
                        a = d @ en["te"]; q = d @ en["ue"]
                        ins = (np.abs(a) < PK_L - PK_TH / 2) & (q > en["rh"] + HM_TH / 2) & (trap == 0)
                        trap[ins] = en["e"]
                if ("press", k) not in done and t >= tp:
                    done.add(("press", k))
                    audio.append((tp, "press", 0.5))
                    tot = int((trap != 0).sum())
                    if not tot:
                        continue
                    grow_lvl = st["t_over"] is not None
                    if grow_lvl:
                        st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                        audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                    audio.append((t, "coinburst", 0.6))
                    for en in ends:
                        sel = trap == en["e"]
                        nt = int(sel.sum())
                        if not nt:
                            continue
                        km = ~sel
                        pos, prev, ang, ghost, trap = pos[km], prev[km], ang[km], ghost[km], trap[km]
                        n = nt * MULT
                        if grow_lvl:
                            capn = int(0.84 * (2 * HS_LEVELS[st["lvl"]]) ** 2 / (math.pi * RB * RB))
                            n = max(nt, min(n, (capn - len(pos)) // 2 + nt))
                        else:
                            n = max(nt, min(n, CAP + 25 - len(pos)))
                        pulses = max(1, min(10, (n + 2) // 3))
                        base, rest = divmod(n, pulses)
                        for p in range(pulses):
                            cnt = base + (1 if p < rest else 0)
                            side = 1 if p % 2 == 0 else -1
                            p0 = C + en["ue"] * (R_P0 + PL - 6) + en["te"] * side * (PK_L + RB + 12)
                            v0 = en["te"] * side * 270 - en["ue"] * 130
                            spawn_q.append([t + p * 0.07, cnt, p0, v0])
                        pops.append((t, en["hm"].copy() + en["ue"] * 30, n))
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, p0, v0 = spawn_q.pop(0)
            if not st["exploded"] and n > 0:
                add(n, p0, v0)
                audio.append((ts, "coin", 0.55))

        if not ex and st["t_over"] is None and len(pos) >= CAP:
            st["t_over"] = t; audio.append((t, "rumble", 0.5))
        if not ex and st["t_over"] is not None:
            st["HS"] += (HS_LEVELS[st["lvl"]] - st["HS"]) * 0.06
            if st["lvl"] >= 3 and t >= st["t_lastpop"] + 1.0:
                st.update(exploded=True, t_expl=t, t_restart=t + 2.6); ex = True
                audio.append((t, "kaching", 1.0)); audio.append((t, "boom", 0.6))
                d = pos - C
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                spd = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / st["HS"])
                prev = pos - (d / dist * spd + rng.normal(0, 120, pos.shape)) * DT
                hs = st["HS"]
                for side in range(4):
                    for q in range(6):
                        s0 = -hs + q * 2 * hs / 6; s1 = s0 + 2 * hs / 6 * 0.9
                        if side == 0: a, b = (C[0] + s0, C[1] - hs), (C[0] + s1, C[1] - hs)
                        elif side == 1: a, b = (C[0] + hs, C[1] + s0), (C[0] + hs, C[1] + s1)
                        elif side == 2: a, b = (C[0] + s0, C[1] + hs), (C[0] + s1, C[1] + hs)
                        else: a, b = (C[0] - hs, C[1] + s0), (C[0] - hs, C[1] + s1)
                        a, b = np.array(a), np.array(b)
                        m = (a + b) / 2
                        dv = (m - C) / np.linalg.norm(m - C)
                        frags.append(dict(a=a - m, b=b - m, c=m, rot=0.0, v=dv * rng.uniform(500, 1100),
                                          w=rng.uniform(-6, 6)))
                for q in range(45):
                    a0 = rng.uniform(0, 2 * math.pi)
                    bills.append(dict(p=C + rng.normal(0, 30, 2), v=np.array([math.cos(a0), math.sin(a0)]) * rng.uniform(300, 900),
                                      r=rng.uniform(0, 360), w=rng.uniform(-400, 400), s=rng.uniform(0.9, 1.5)))
        if ex and t >= st["t_restart"]:
            st.update(exploded=False, t_over=None, HS=HS0, lvl=0, t_lastpop=None, restart0=t); ex = False
            frags, bills = [], []
            nxt = [k for k, tp in enumerate(PRESS_T) if tp > t + 1.2]
            place_single(PRESS_T[nxt[0]])
            for k, tp in enumerate(PRESS_T):
                if k < nxt[0]:
                    done.add(("trap", k)); done.add(("press", k))
            audio.append((t, "coin", 0.6))

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
                v += rel / HS0 * OUTF * DT
                v[ghost] = 0
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            cap = 2500 if ex else 700
            v = np.where(spd > cap, v / np.maximum(spd, 1e-6) * cap, v)
            prev = pos; pos = pos + v * DT + grav * DT * DT
            ang = ang + v[:, 0] * DT / RB * 0.35
            _, _, _, ends_s = mech_at(ts)
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
                        sel = trap == en["e"]
                        if sel.any():
                            d = pos[sel] - C
                            a = d @ en["te"]; q = d @ en["ue"]
                            a = np.clip(a, -(PK_L - PK_TH / 2 - RB), PK_L - PK_TH / 2 - RB)
                            q = np.maximum(q, en["rh"] + HM_TH / 2 + RB * 0.6)
                            pos[sel] = C + a[:, None] * en["te"] + q[:, None] * en["ue"]
                    dg = pos - C
                    dd = np.linalg.norm(dg, axis=1)
                    hg = solid & (dd < RB + GEAR_R)
                    if hg.any():
                        pos[hg] = C + dg[hg] / np.maximum(dd[hg], 1e-4)[:, None] * (RB + GEAR_R)
                    lim = st["HS"] - RB
                    for ax in (0, 1):
                        lo = pos[:, ax] < C[ax] - lim
                        hi = pos[:, ax] > C[ax] + lim
                        for msk, val in ((lo, C[ax] - lim), (hi, C[ax] + lim)):
                            if msk.any():
                                vv = pos[msk, ax] - prev[msk, ax]
                                pos[msk, ax] = val
                                prev[msk, ax] = val + vv * 0.85
        if ex:
            kp = (pos[:, 0] > -50) & (pos[:, 0] < W + 50) & (pos[:, 1] < H + 50) & (pos[:, 1] > -500)
            pos, prev, ang, ghost, trap = pos[kp], prev[kp], ang[kp], ghost[kp], trap[kp]
            for fr in frags:
                fr["c"] = fr["c"] + fr["v"] / FPS; fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS
                fr["rot"] += fr["w"] / FPS
            for b in bills:
                b["p"] = b["p"] + b["v"] / FPS; b["v"] = b["v"] * 0.97 + np.array([0, 500.0]) / FPS
                b["r"] += b["w"] / FPS
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), ang=ang.copy(), HS=st["HS"], exploded=ex,
                               t_over=st["t_over"], lvl=st["lvl"], t_lastpop=st["t_lastpop"],
                               t_expl=st["t_expl"], restart0=st["restart0"],
                               frags=[dict(f) for f in frags], bills=[dict(b) for b in bills]))
    return frames, audio, pops, st


# ---------------- desenho ----------------
EMOJI_FONT = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)


def emoji_rgba(ch, d):
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=EMOJI_FONT, embedded_color=True)
    im = im.crop(im.getbbox())
    s = max(im.size)
    sq = Image.new("RGBA", (s, s), (0, 0, 0, 0)); sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return sq


BASE = emoji_rgba("\U0001F911", 0)
NROT = 24


def make_sprites(img, d, nrot):
    out = []
    big = img.resize((d * 4, d * 4), Image.LANCZOS)
    for r in range(nrot):
        rr = big.rotate(-360 * r / nrot, resample=Image.BICUBIC).resize((d, d), Image.LANCZOS)
        a = np.asarray(rr).astype(np.float32)
        al = a[..., 3:4] / 255
        bgr = a[..., [2, 1, 0]]
        out.append(((bgr * al).astype(np.float32), (1 - al).astype(np.float32)))
    return out


SPR = make_sprites(BASE, SPR_D, NROT)
SS = SPR_D; HALF = SS // 2
BILL = emoji_rgba("\U0001F4B5", 0)
SH = 16


def P(p):
    return (int(round(p[0] * SH)), int(round(p[1] * SH)))


def blit(f, pre, inv, x0, y0):
    s = pre.shape[0]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if xa >= xb or ya >= yb:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * inv[sl] + pre[sl]


def make_glow(r):
    s = int(r * 2)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    return np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 2


GLOW = make_glow(150)


def add_glow(img, center, color, k):
    s = GLOW.shape[0]
    x0, y0 = int(center[0] - s / 2), int(center[1] - s / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if xa >= xb or ya >= yb or k <= 0.01:
        return
    gg = GLOW[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * k
    reg = img[ya:yb, xa:xb].astype(np.float32) + gg * np.array(color, np.float32)
    img[ya:yb, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)


def gear_poly(center, a0, r, teeth):
    pts = []
    for i in range(teeth * 4):
        a = a0 + i * (2 * math.pi / (teeth * 4))
        rr = r if (i % 4) in (0, 1) else r * 0.82
        pts.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a)))
    return (np.array(pts) * SH).astype(np.int32)


def draw_rod(img, a, b, th, base=(165, 165, 170)):
    d = b - a; L = float(np.linalg.norm(d)); u = d / L; n = np.array([-u[1], u[0]]); h = th / 2
    cv2.line(img, P(a), P(b), (70, 70, 70), int(th + 3), cv2.LINE_AA, 4)
    cv2.line(img, P(a), P(b), base, int(th), cv2.LINE_AA, 4)
    cv2.line(img, P(a + n * (h - 3)), P(b + n * (h - 3)), (215, 215, 220), 2, cv2.LINE_AA, 4)
    cv2.line(img, P(a - n * (h - 3)), P(b - n * (h - 3)), (115, 115, 120), 2, cv2.LINE_AA, 4)


def square_pts(hs, wob, t):
    pts = []
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for i in range(4):
        a = np.array(corners[i], float); b = np.array(corners[(i + 1) % 4], float)
        nrm = np.array([0.0, -1.0]) if i == 0 else np.array([1.0, 0]) if i == 1 else np.array([0, 1.0]) if i == 2 else np.array([-1.0, 0])
        for s in np.linspace(0, 1, 40, endpoint=False):
            p = C + (a + (b - a) * s) * hs
            bump = math.sin(math.pi * s) * wob * (0.7 + 0.3 * math.sin(t * 25 + i * 1.7 + s * 9))
            pts.append(p + nrm * bump)
    return np.array(pts)


def render(fr, pops, srng):
    t = fr["t"]; ex = fr["exploded"]
    img = np.zeros((H, W, 3), np.uint8)
    act = []
    for (t0, gc, n) in pops:
        if t0 - 0.02 <= t < t0 + 0.9:
            k = 1.0 if t < t0 + 0.12 else max(0.0, 1 - (t - t0 - 0.12) / 0.78)
            act.append((gc, n, k, t - t0)); add_glow(img, gc, (60, 230, 120), k * 0.8)
    f = img.astype(np.float32)
    ip = np.round(fr["pos"]).astype(np.int32)
    ri = (np.round(fr["ang"] / (2 * math.pi) * NROT).astype(int)) % NROT
    for k in range(len(ip)):
        pre, inv = SPR[ri[k]]
        x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
        if 0 <= x0 and 0 <= y0 and x0 + SS <= W and y0 + SS <= H:
            reg = f[y0:y0 + SS, x0:x0 + SS]; reg *= inv; reg += pre
        else:
            blit(f, pre, inv, x0, y0)
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
        pts = square_pts(fr["HS"], wob, t)
        cv2.polylines(ov, [(pts * SH).astype(np.int32)], True, GOLD, 4, cv2.LINE_AA, 4)
        img = ov if fade >= 1 else cv2.addWeighted(ov, fade, img, 1 - fade, 0)
    for fg in fr["frags"]:
        cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"])
        R = np.array([[cr, -sr], [sr, cr]])
        a = fg["c"] + R @ fg["a"]; b = fg["c"] + R @ fg["b"]
        cv2.line(img, P(a), P(b), GOLD, 4, cv2.LINE_AA, 4)

    ta = 1.0
    if ex:
        ta = max(0.0, 1 - (t - fr["t_expl"]) / 0.25)
    elif r0 is not None:
        ta = fade
    if ta > 0:
        th, u, tv, ends = mech_at(t)
        ov = img.copy()
        # sombra do bastão (ele passa por cima dos emojis)
        sh_off = np.array([5.0, 7.0])
        cv2.line(ov, P(ends[0]["B"] + sh_off), P(ends[1]["B"] + sh_off), (12, 12, 12), int(ROD_TH + 4), cv2.LINE_AA, 4)
        for en in ends:
            ue, te, B = en["ue"], en["te"], en["B"]
            # pistão + martelo
            draw_rod(ov, B, en["hm"], 8, base=(120, 120, 125))
            hm_a, hm_b = en["hm"] - te * (PK_L - 12), en["hm"] + te * (PK_L - 12)
            draw_rod(ov, hm_a, hm_b, HM_TH, base=(70, 170, 220))
            # suporte em "U" (bastão dobrado)
            c1, c2 = B - te * PK_L, B + te * PK_L
            draw_rod(ov, c1, c2, PK_TH)
            draw_rod(ov, c1, c1 + ue * PL, PK_TH)
            draw_rod(ov, c2, c2 + ue * PL, PK_TH)
            for cc in (c1, c2):
                cv2.circle(ov, P(cc), int(PK_TH / 2 * SH), (165, 165, 170), -1, cv2.LINE_AA, 4)
                cv2.circle(ov, P(cc), int(3 * SH), (80, 80, 80), -1, cv2.LINE_AA, 4)
        draw_rod(ov, ends[0]["B"], ends[1]["B"], ROD_TH)
        # engrenagem central (gira com o bastão) + pinhão
        gp = gear_poly(C, th, GEAR_R, 16)
        cv2.fillPoly(ov, [gp], (70, 175, 215), cv2.LINE_AA, 4)
        cv2.polylines(ov, [gp], True, (30, 90, 120), 1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(C), int(GEAR_R * 0.62 * SH), (40, 120, 160), 2, cv2.LINE_AA, 4)
        for q in range(6):
            a = th + q * math.pi / 3
            cv2.circle(ov, P(C + 0.42 * GEAR_R * np.array([math.cos(a), math.sin(a)])), int(5 * SH), (30, 100, 140), -1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(C), int(13 * SH), (190, 190, 195), -1, cv2.LINE_AA, 4)
        (tw, tht), _ = cv2.getTextSize("$", cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(ov, "$", (int(C[0] - tw / 2), int(C[1] + tht / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.75, (40, 140, 40), 2, cv2.LINE_AA)
        img = ov if ta >= 1 else cv2.addWeighted(ov, ta, img, 1 - ta, 0)

    # "+N" subindo
    for (gc, n, k, dtt) in act:
        if dtt >= 0:
            txt = f"+{n}"
            p = gc - (gc - C) / max(1e-3, np.linalg.norm(gc - C)) * (40 + dtt * 60)
            al = min(1.0, k * 1.3)
            ov = img.copy()
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            org = (int(p[0] - tw / 2), int(p[1] + tht / 2))
            cv2.putText(ov, txt, org, cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 60, 20), 5, cv2.LINE_AA)
            cv2.putText(ov, txt, org, cv2.FONT_HERSHEY_DUPLEX, 0.9, (90, 240, 120), 2, cv2.LINE_AA)
            img = cv2.addWeighted(ov, al, img, 1 - al, 0)

    if ex:
        f = img.astype(np.float32)
        for b in fr["bills"]:
            d = int(30 * b["s"])
            key = (d, int(b["r"]) % 360 // 15)
            if key not in BILL_CACHE:
                rr = BILL.resize((d * 3, d * 3), Image.LANCZOS).rotate(key[1] * 15, resample=Image.BICUBIC, expand=False).resize((d, d), Image.LANCZOS)
                a = np.asarray(rr).astype(np.float32); al = a[..., 3:4] / 255
                BILL_CACHE[key] = ((a[..., [2, 1, 0]] * al).astype(np.float32), (1 - al).astype(np.float32))
            pre, inv = BILL_CACHE[key]
            blit(f, pre, inv, int(b["p"][0]) - d // 2, int(b["p"][1]) - d // 2)
        img = np.clip(f, 0, 255).astype(np.uint8)
        fl = max(0.0, 0.85 * (1 - (t - fr["t_expl"]) / 0.35))
        if fl > 0:
            img = cv2.addWeighted(img, 1 - fl, np.full_like(img, (200, 255, 230)), fl, 0)
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


BILL_CACHE = {}

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "search":
        for seed in range(int(sys.argv[2]), int(sys.argv[3])):
            _, au, pops, st = simulate(seed, keep=False)
            full = [a[0] for a in au if a[1] == "rumble"]
            boom = [a[0] for a in au if a[1] == "kaching"]
            print(seed, "cheio", round(full[0], 1) if full else None, "explode", [round(b, 1) for b in boom],
                  "aperto", [(round(p[0], 1), p[2]) for p in pops][:10], flush=True)
        sys.exit()
    seed = int(sys.argv[2])
    frames, audio, pops, st = simulate(seed)
    r0 = st["restart0"]
    if r0 is not None:
        later = [p[0] for p in pops if p[0] > r0]
        t_end = (later[0] + 1.3) if later else r0 + 3.5
    else:
        t_end = DUR
    frames = [f for f in frames if f["t"] < t_end]
    audio = [a for a in audio if a[0] < t_end]
    print("duracao", len(frames) / FPS, "explosao", st["t_expl"], flush=True)
    srng = np.random.default_rng(1)
    if mode == "preview":
        for fi in [int(x) for x in sys.argv[3:]]:
            if fi < len(frames):
                cv2.imwrite(f"out/d_{fi:04d}.png", render(frames[fi], pops, srng))
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "out/d_video_tmp.mp4"], stdin=subprocess.PIPE)
        for fr in frames:
            enc.stdin.write(render(fr, pops, srng).tobytes())
        enc.stdin.close(); enc.wait()
        json.dump(dict(dur=len(frames) / FPS, events=sorted(audio)), open("out/d_audio_events.json", "w"))
        print("ok", enc.returncode)
