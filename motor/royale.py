"""BATALHA ROYALE — QUAL PAÍS SOBREVIVE?
8 times numa arena. Rotor com caixas que abocanham e esmagam: quem encosta no martelo leva o bolso.
A cada 5 s o RELÓGIO DE ELIMINAÇÃO zera e o time com menos bolinhas é ELIMINADO (estouram).
Ranking ao vivo (corrida de barras), FINAL com 2, tela de CAMPEÃO."""
import sys, os, math, json, subprocess, wave
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena as A
from arena import (W, H, FPS, SUB, DT, T0, CYC, SH, P, mech_at, push_capsule, emoji_img, to_pre, blit,
                   emoji_scaled, put_text, text_img, add_glow, gear_poly, draw_rod, ease_out_back, Conf, _cache)

TEAMS = [  # (emoji, nome, cor BGR, deslocamento x do recorte)
    ("\U0001F1E7\U0001F1F7", "BRASIL", (60, 190, 40), 0.0),
    ("\U0001F1E6\U0001F1F7", "ARGENTINA", (235, 195, 120), 0.0),
    ("\U0001F1F5\U0001F1F9", "PORTUGAL", (50, 50, 200), -0.12),
    ("\U0001F1FA\U0001F1F8", "EUA", (150, 60, 40), -0.2),
    ("\U0001F1EB\U0001F1F7", "FRANÇA", (230, 120, 60), 0.0),
    ("\U0001F1EF\U0001F1F5", "JAPÃO", (230, 230, 240), 0.0),
    ("\U0001F1E9\U0001F1EA", "ALEMANHA", (20, 200, 250), 0.0),
    ("\U0001F1EA\U0001F1F8", "ESPANHA", (30, 110, 245), -0.15),
]
NT = len(TEAMS)
GENDER = {"BRASIL": "o", "ARGENTINA": "a", "PORTUGAL": "o", "EUA": "os", "FRANÇA": "a", "JAPÃO": "o", "ALEMANHA": "a", "ESPANHA": "a"}


def elim_word(name):
    return {"o": "ELIMINADO", "a": "ELIMINADA", "os": "ELIMINADOS"}[GENDER.get(name, "o")]


def champ_word(name):
    return {"o": "CAMPEÃO", "a": "CAMPEÃ", "os": "CAMPEÕES"}[GENDER.get(name, "o")]
TITLE = "QUAL PAÍS SOBREVIVE?"
ELIM_T = [7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 42.0]
END_T = 47.6
INIT = 30
SOFT_CAP = 360
CX, CY, RAD = 360.0, 660.0, 262.0
CFG = dict(shape="circle", pts=[(CX + RAD * math.cos(a), CY + RAD * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 180, endpoint=False)],
           arms=4, names=("", ""), emo=("", ""), col=((0, 0, 0), (0, 0, 0)), snd=("pop", "pop"), bg=(36, 16, 12), RB=11.0)


# ======================= simulação =======================
def simulate(ar, seed, keep=True, swirl=-0.35):
    rng = np.random.default_rng(seed)
    RB = ar.RB; C = ar.C
    # posições iniciais espalhadas no anel por onde as caixas passam
    n0 = INIT * NT
    ang0 = rng.uniform(0, 2 * math.pi, n0); rr0 = np.sqrt(rng.uniform((ar.R1 / RAD) ** 2, ((RAD - RB - 4) / RAD) ** 2, n0)) * RAD
    pos = C + np.stack([np.cos(ang0) * rr0, np.sin(ang0) * rr0], 1)
    prev = pos.copy(); ang = rng.uniform(-3, 3, n0)
    team = np.repeat(np.arange(NT), INIT).astype(np.int8)
    rng.shuffle(team)
    trap = np.zeros(n0, np.int8)
    alive = [True] * NT
    frames, audio, pops, elims, parts = [], [], [], [], []
    spawn_q, done = [], set()
    st = dict(champion=None, t_end=None)
    NF = int(END_T * FPS)
    audio.append((0.1, "gong", 0.8))
    t_tick = 0.5; early = 0

    def add(n, p0, v0, tm):
        nonlocal pos, prev, ang, team, trap
        p = p0 + rng.normal(0, 4, (n, 2)); v = v0 + rng.normal(0, 60, (n, 2))
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        ang = np.concatenate([ang, rng.uniform(-0.5, 0.5, n)])
        team = np.concatenate([team, np.full(n, tm, np.int8)]); trap = np.concatenate([trap, np.zeros(n, np.int8)])

    def eliminate(tm, t):
        nonlocal pos, prev, ang, team, trap
        alive[tm] = False
        sel = team == tm
        parts.append((t, pos[sel].copy(), tm))
        km = ~sel
        pos, prev, ang, team, trap = pos[km], prev[km], ang[km], team[km], trap[km]
        elims.append((t, tm, sum(alive)))
        audio.append((t, "elim", 1.0))
        if sum(alive) == 2:
            audio.append((t + 1.0, "final", 0.9))

    ei = 0
    for fi in range(NF):
        t = fi / FPS
        while t_tick <= t:
            audio.append((t_tick, "tick", 0.08)); t_tick += 0.5
        th, k, u, beta, boxes = mech_at(ar, t)
        champ = st["champion"] is not None

        if k >= 0 and not champ:
            tc = T0 + k * CYC
            if ("trap", k) not in done and u >= 0.35:
                done.add(("trap", k))
                audio.append((tc + 0.35, "clack", 0.25))
                for b in boxes:
                    d = pos - b["B"]; x = d @ b["f"]; y = d @ b["ue"]
                    ins = (x > b["hx"] + 7) & (x < ar.D - 3) & (np.abs(y) < ar.Wb / 2 - ar.PK / 2) & (trap == 0)
                    trap[ins] = b["i"] + 1
            if ("press", k) not in done and u >= 0.70:
                done.add(("press", k))
                audio.append((tc + 0.70, "press", 0.35))
                for b in boxes:
                    sel = trap == b["i"] + 1
                    nt = int(sel.sum())
                    if not nt:
                        continue
                    xs = (pos[sel] - b["B"]) @ b["f"]
                    w = int(team[sel][int(np.argmin(xs))])
                    mixed = len(set(team[sel].tolist())) > 1
                    km = ~sel
                    pos, prev, ang, team, trap = pos[km], prev[km], ang[km], team[km], trap[km]
                    grow = 1.6 if len(pos) + nt < SOFT_CAP else 1.0
                    n = max(1, int(round(nt * grow)))
                    if mixed:
                        audio.append((t + 0.02, "steal", 0.5))
                    pulses = max(1, min(6, (n + 2) // 3)); base, rest = divmod(n, pulses)
                    for p in range(pulses):
                        spawn_q.append([t + 0.06 + p * 0.07, base + (1 if p < rest else 0), b["i"], 1 if p % 2 == 0 else -1, w])
                    pops.append((t, (b["B"] + b["f"] * ar.D * 0.5).tolist(), n, w, mixed))
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, bi, side, tm = spawn_q.pop(0)
            if n > 0 and alive[tm] and not champ:
                b = boxes[bi]
                p0 = b["B"] + b["f"] * ar.D * 0.5 + b["ue"] * side * (ar.Wb / 2 + RB + 8)
                v0 = b["ue"] * side * 230 + b["f"] * 60
                if not ar.inside(p0[None])[0]:
                    p0 = b["B"] + b["f"] * ar.D * 0.5 - b["ue"] * (ar.Wb / 2 + RB + 8); v0 = -b["ue"] * 230
                add(n, p0, v0, tm)
                audio.append((ts, "n%d" % tm, 0.45))

        # zerou antes da hora -> eliminado na hora
        if not champ:
            for tm in range(NT):
                if alive[tm] and not (team == tm).any() and not any(q[4] == tm for q in spawn_q):
                    eliminate(tm, t); early += 1
            # relógio de eliminação
            if ei < len(ELIM_T) and t >= ELIM_T[ei]:
                target_alive = NT - 1 - ei
                while sum(alive) > max(1, target_alive):
                    cnt = [(int((team == tm).sum()), rng.uniform(), tm) for tm in range(NT) if alive[tm]]
                    eliminate(min(cnt)[2], t)
                ei += 1
            # countdown beeps
            if ei < len(ELIM_T):
                for s_ in (3, 2, 1):
                    tb = ELIM_T[ei] - s_
                    if ("beep", ei, s_) not in done and t >= tb:
                        done.add(("beep", ei, s_)); audio.append((tb, "beep", 0.6))
            if sum(alive) == 1:
                st["champion"] = alive.index(True); st["t_end"] = t
                audio.append((t + 0.3, "fanfare", 0.9)); audio.append((t + 0.3, "crowd", 0.8)); audio.append((t, "boom", 0.6))

        for s in range(SUB):
            if len(pos) == 0:
                break
            ts = t + s * DT
            v = (pos - prev) / DT * 0.9993
            rel = pos - C
            vt = np.stack([-rel[:, 1], rel[:, 0]], 1) * swirl
            v += (vt - v) * min(1.0, 0.7 * DT)
            rr = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-6
            v -= (rel / rr) * np.clip((rr - ar.Rm) / ar.r_in, -1, 1) * 90 * DT
            if champ:
                v *= 0.97
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(spd > 650, v / np.maximum(spd, 1e-6) * 650, v)
            prev = pos; pos = pos + v * DT
            ang = ang + v[:, 0] * DT / RB * 0.35
            _, _, _, _, bxs = mech_at(ar, ts)
            for it in range(2):
                pr = cKDTree(pos).query_pairs(2 * RB, output_type="ndarray")
                if len(pr):
                    i, j = pr[:, 0], pr[:, 1]
                    d = pos[j] - pos[i]
                    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4)
                    corr = d * ((2 * RB - dist) / dist * 0.5 * 0.85)[:, None]
                    dp = np.zeros_like(pos); np.add.at(dp, i, -corr); np.add.at(dp, j, corr)
                    pos += dp
                free = trap == 0
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
                dg = pos - C; dd = np.linalg.norm(dg, axis=1)
                hg = dd < RB + ar.GEAR
                if hg.any():
                    pos[hg] = C + dg[hg] / np.maximum(dd[hg], 1e-4)[:, None] * (RB + ar.GEAR)
                sd, gx, gy = ar.sample(pos, 1.0)
                o = sd < RB
                if o.any():
                    g = np.stack([gx[o], gy[o]], 1)
                    pos[o] += g * (RB - sd[o])[:, None]
                    vv = pos[o] - prev[o]; vn = (vv * g).sum(1)
                    prev[o] = pos[o] - (vv - np.minimum(vn, 0)[:, None] * g * 1.85)
        counts = [int((team == tm).sum()) for tm in range(NT)]
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), ang=ang.copy(), team=team.copy(), counts=counts, alive=list(alive),
                               ei=ei, champion=st["champion"], t_end=st["t_end"]))
        if st["t_end"] is not None and t > st["t_end"] + 5.2:
            break
    return frames, audio, pops, elims, parts, st, early


def score(elims, st, early):
    if st["champion"] is None:
        return -1e9, "sem campeão"
    final = [e[1] for e in elims][-1:] + [st["champion"]]
    sc = -early * 6
    names = [TEAMS[i][1] for i in final]
    if set(final) == {0, 1}:
        sc += 8
    elif 0 in final:
        sc += 3
    return sc, f"eliminados {[TEAMS[e[1]][1] + '@' + str(round(e[0], 1)) for e in elims]} campeão {TEAMS[st['champion']][1]} precoce {early}"


# ======================= desenho =======================
def flag_badge(ch, d, xoff=0.0):
    im = emoji_img(ch)
    w, h = im.size
    bb = im.getbbox()
    im = im.crop(bb); w, h = im.size
    side = int(min(w, h) * 0.86)
    cx = w / 2 + xoff * w; cy = h / 2
    x0 = int(max(0, min(w - side, cx - side / 2))); y0 = int(max(0, min(h - side, cy - side / 2)))
    sq = im.crop((x0, y0, x0 + side, y0 + side)).resize((d * 4, d * 4), Image.LANCZOS)
    mask = Image.new("L", (d * 4, d * 4), 0)
    ImageDraw.Draw(mask).ellipse((6, 6, d * 4 - 6, d * 4 - 6), fill=255)
    out = Image.new("RGBA", (d * 4, d * 4), (0, 0, 0, 0))
    out.paste(sq, (0, 0), mask)
    ImageDraw.Draw(out).ellipse((6, 6, d * 4 - 6, d * 4 - 6), outline=(255, 255, 255, 255), width=max(3, d // 5))
    return out


class Renderer:
    def __init__(self, ar):
        _cache.clear()
        self.ar = ar
        d = int(round(ar.RB * 2.25)); self.SD = d; self.HALF = d // 2
        self.NROT = 12
        self.BADGE = [flag_badge(e, 64, xo) for (e, n, c, xo) in TEAMS]
        self.SPR = [[to_pre(b.rotate(-360 * r / self.NROT, resample=Image.BICUBIC).resize((d, d), Image.LANCZOS)) for r in range(self.NROT)]
                    for b in self.BADGE]
        self.FLAG = [emoji_img(e) for (e, n, c, xo) in TEAMS]
        self.CROWN = emoji_img("\U0001F451"); self.FINGER = emoji_img("\U0001F447"); self.CLOCK = emoji_img("⏱️")
        self.X = emoji_img("❌"); self.TROPHY = emoji_img("\U0001F3C6")
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dd = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2) / 760
        g = np.clip(1 - dd, 0, 1) ** 1.5
        bg = np.zeros((H, W, 3), np.float32); bg[:] = (10, 6, 6)
        bg += g[..., None] * np.array([60, 28, 18], np.float32)
        fl = np.zeros((H, W), np.float32); fl[ar.mask > 0] = 1
        fl = cv2.GaussianBlur(fl, (0, 0), 1.0)[..., None]
        self.BG = bg * (1 - fl * 0.6) + fl * 0.6 * np.array([40, 26, 22], np.float32)
        self.rowy = {tm: 0.0 for tm in range(NT)}
        self.first = True

    def leaderboard(self, f, fr, t):
        counts, alive = fr["counts"], fr["alive"]
        order = sorted(range(NT), key=lambda tm: (not alive[tm], -counts[tm] if alive[tm] else 0, tm))
        y0, rh = 92, 35
        mx = max(1, max(counts))
        nalive = sum(alive)
        last_alive = order[nalive - 1] if nalive > 1 else None
        to_elim = fr["ei"] < len(ELIM_T) and fr["champion"] is None
        tleft = (ELIM_T[fr["ei"]] - t) if to_elim else 99
        for rank, tm in enumerate(order):
            ty = y0 + rank * rh
            if self.first:
                self.rowy[tm] = ty
            self.rowy[tm] += (ty - self.rowy[tm]) * 0.22
        self.first = False
        img = np.clip(f, 0, 255).astype(np.uint8)
        for rank, tm in enumerate(order):
            y = int(self.rowy[tm])
            col = TEAMS[tm][2]
            if not alive[tm]:
                cv2.rectangle(img, (40, y - 15), (680, y + 15), (28, 24, 26), -1, cv2.LINE_AA)
                continue
            danger = (tm == last_alive and tleft <= 3.0)
            base = (40, 30, 34)
            if danger:
                k = 0.5 + 0.5 * math.sin(t * 18)
                base = (int(40 + 20 * k), int(30 + 10 * k), int(120 + 110 * k))
            cv2.rectangle(img, (40, y - 15), (680, y + 15), base, -1, cv2.LINE_AA)
            bw = int(330 * counts[tm] / mx)
            cv2.rectangle(img, (300, y - 9), (300 + max(2, bw), y + 9), col, -1, cv2.LINE_AA)
        f = img.astype(np.float32)
        for rank, tm in enumerate(order):
            y = self.rowy[tm]
            al = 1.0 if alive[tm] else 0.45
            if alive[tm]:
                put_text(f, f"{rank + 1}º", 20, 66, y, (220, 220, 220), stroke=2, stroke_col=(20, 15, 15), alpha=al)
            blit(f, *emoji_scaled(self.FLAG[tm], 30), 90, int(y - 15), al)
            pre, inv = text_img(TEAMS[tm][1], 22, (255, 255, 255) if alive[tm] else (140, 140, 140), 2, (20, 15, 15))
            blit(f, pre, inv, 130, int(y - pre.shape[0] / 2), al)
            if alive[tm]:
                put_text(f, str(fr["counts"][tm]), 22, 300 + int(330 * counts[tm] / mx) + 26, y, (255, 255, 255), stroke=2, stroke_col=(20, 15, 15))
            else:
                blit(f, *emoji_scaled(self.X, 24), 640, int(y - 12), 0.9)
        return f

    def render(self, fr, pops, elims, parts, st, srng, conf):
        ar = self.ar; C = ar.C; t = fr["t"]
        f = self.BG.copy()
        for (t0, gc, n, tm, mixed) in pops:
            if t0 - 0.02 <= t < t0 + 0.8:
                k = 1.0 if t < t0 + 0.12 else max(0.0, 1 - (t - t0 - 0.12) / 0.68)
                add_glow(f, gc, TEAMS[tm][2], k * 0.6, 100)
        ip = np.round(fr["pos"]).astype(np.int32)
        ri = (np.round(fr["ang"] / (2 * math.pi) * self.NROT).astype(int)) % self.NROT
        SD, HALF = self.SD, self.HALF
        for k in range(len(ip)):
            pre, inv = self.SPR[fr["team"][k]][ri[k]]
            x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
            if 0 <= x0 and 0 <= y0 and x0 + SD <= W and y0 + SD <= H:
                reg = f[y0:y0 + SD, x0:x0 + SD]; reg *= inv; reg += pre
            else:
                blit(f, pre, inv, x0, y0)
        img = np.clip(f, 0, 255).astype(np.uint8)
        # estilhaços dos eliminados
        for (t0, ps, tm) in parts:
            dtt = t - t0
            if 0 <= dtt < 0.8:
                col = TEAMS[tm][2]
                r0 = np.random.default_rng(int(t0 * 100) + tm)
                for p in ps:
                    for q in range(5):
                        a = r0.uniform(0, 2 * math.pi); sp = r0.uniform(80, 260)
                        pp = p + np.array([math.cos(a), math.sin(a)]) * sp * dtt
                        rr = max(1, int(5 * (1 - dtt / 0.8)))
                        cv2.circle(img, (int(pp[0]), int(pp[1])), rr, col, -1, cv2.LINE_AA)
                if dtt < 0.4:
                    for p in ps[::3]:
                        cv2.circle(img, (int(p[0]), int(p[1])), int(10 + 40 * dtt / 0.4), (255, 255, 255), 2, cv2.LINE_AA)
        # borda
        lead = int(np.argmax(fr["counts"]))
        glow = np.zeros_like(img)
        pts = (ar.dense_outline(1.0, 0, t) * SH).astype(np.int32)
        cv2.polylines(glow, [pts], True, TEAMS[lead][2], 14, cv2.LINE_AA, 4)
        img = cv2.add(img, (cv2.GaussianBlur(glow, (0, 0), 9) * 0.85).astype(np.uint8))
        cv2.polylines(img, [pts], True, (245, 240, 250), 4, cv2.LINE_AA, 4)
        # rotor
        th, k, u, beta, boxes = mech_at(ar, t)
        champ = fr["champion"] is not None
        ta = 1.0 if not champ else max(0.0, 1 - (t - fr["t_end"]) / 0.4)
        if ta > 0:
            ov = img.copy()
            for b in boxes:
                cv2.line(ov, P(C + np.array([5.0, 7.0])), P(b["B"] + np.array([5.0, 7.0])), (6, 4, 6), 19, cv2.LINE_AA, 4)
            for b in boxes:
                draw_rod(ov, C, b["B"], 14)
                draw_rod(ov, b["B"], b["hm"], 7, base=(120, 115, 120))
                a_, b_, th_ = b["segs"][5]; draw_rod(ov, a_, b_, th_, base=(70, 180, 235))
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
            cv2.circle(ov, P(C), int(10 * SH), (200, 195, 200), -1, cv2.LINE_AA, 4)
            img = ov if ta >= 1 else cv2.addWeighted(ov, ta, img, 1 - ta, 0)

        f = img.astype(np.float32)
        put_text(f, TITLE, 40, W / 2, 46, (255, 255, 255), stroke=4, stroke_col=(30, 20, 60))
        f = self.leaderboard(f, fr, t)

        # relógio
        ei = fr["ei"]
        nalive = sum(fr["alive"])
        if not champ and ei < len(ELIM_T):
            tl = max(0.0, ELIM_T[ei] - t)
            sec = int(math.ceil(tl - 1e-6))
            red = tl <= 3.0
            col = (60, 60, 255) if red else (255, 255, 255)
            sc = 1.0 + (0.25 * max(0, 1 - (math.ceil(tl) - tl) * 4) if red else 0)
            label = "FINAL" if nalive == 2 else "ELIMINAÇÃO EM"
            put_text(f, label, 34, W / 2 + 10, 965, (255, 255, 255), stroke=3, stroke_col=(30, 20, 30))
            blit(f, *emoji_scaled(self.CLOCK, 52), 228 if nalive > 2 else 270, 1000)
            put_text(f, f"{sec}s", 62, W / 2 + 30, 1028, col, stroke=5, stroke_col=(30, 20, 30), scale=sc)
            # barra do tempo
            span = ELIM_T[ei] - (ELIM_T[ei - 1] if ei > 0 else 2.0)
            frac = min(1.0, tl / span)
            img = np.clip(f, 0, 255).astype(np.uint8)
            cv2.rectangle(img, (160, 1072), (560, 1082), (60, 50, 55), -1, cv2.LINE_AA)
            cv2.rectangle(img, (160, 1072), (160 + int(400 * frac), 1082), col, -1, cv2.LINE_AA)
            f = img.astype(np.float32)
            put_text(f, "Comenta seu país", 30, W / 2 - 24, 1135, (255, 255, 255), stroke=3, stroke_col=(30, 20, 30))
            blit(f, *emoji_scaled(self.FINGER, 44), int(W / 2 + 128), int(1112 + 5 * math.sin(t * 6)))
            put_text(f, f"RESTAM {nalive}", 24, W / 2, 1186, (200, 200, 210), stroke=2, stroke_col=(30, 20, 30))
        # intro
        if t < 2.2:
            al = 1.0 if t < 1.7 else max(0.0, 1 - (t - 1.7) / 0.5)
            img = np.clip(f, 0, 255).astype(np.uint8)
            ov = img.copy(); cv2.rectangle(ov, (40, 560), (680, 760), (25, 15, 20), -1, cv2.LINE_AA)
            img = cv2.addWeighted(ov, 0.75 * al, img, 1 - 0.75 * al, 0); f = img.astype(np.float32)
            sc = ease_out_back(min(1.0, t / 0.35)) if t > 0 else 0.05
            put_text(f, "ESCOLHA SEU PAÍS", 50, W / 2, 625, (80, 230, 255), stroke=5, stroke_col=(30, 20, 30), alpha=al, scale=max(0.05, sc))
            put_text(f, "e assista até o fim!", 34, W / 2, 695, (255, 255, 255), stroke=3, stroke_col=(30, 20, 30), alpha=al)
        # banner de eliminação
        for (t0, tm, rest) in elims:
            dtt = t - t0
            if 0 <= dtt < 1.6 and not (champ and rest == 1):
                al = 1.0 if dtt < 1.2 else max(0.0, 1 - (dtt - 1.2) / 0.4)
                sc = ease_out_back(min(1.0, dtt / 0.3))
                img = np.clip(f, 0, 255).astype(np.uint8)
                ov = img.copy(); cv2.rectangle(ov, (30, 560), (690, 770), (20, 10, 60), -1, cv2.LINE_AA)
                img = cv2.addWeighted(ov, 0.8 * al, img, 1 - 0.8 * al, 0); f = img.astype(np.float32)
                d = int(90 * max(0.05, sc))
                blit(f, *emoji_scaled(self.FLAG[tm], d), int(W / 2 - d / 2), int(612 - d / 2), al)
                put_text(f, f"{TEAMS[tm][1]} {elim_word(TEAMS[tm][1])}!", 48, W / 2, 690, (70, 70, 255), stroke=5, stroke_col=(255, 255, 255), alpha=al, scale=max(0.05, sc), maxw=640)
                txt = "FINAL!" if rest == 2 else f"RESTAM {rest}"
                put_text(f, txt, 32, W / 2, 742, (255, 255, 255), stroke=3, stroke_col=(30, 20, 30), alpha=al)
        img = np.clip(f, 0, 255).astype(np.uint8)
        # campeão
        if champ:
            te = t - fr["t_end"]
            k = min(1.0, te / 0.5)
            img = (img.astype(np.float32) * (1 - 0.7 * k)).astype(np.uint8)
            conf.draw(img, t)
            f = img.astype(np.float32)
            wv = fr["champion"]
            sc = ease_out_back(min(1.0, (te - 0.2) / 0.45)) if te > 0.2 else 0
            if sc > 0.02:
                d = int(260 * sc * (1 + 0.03 * math.sin(te * 6)))
                add_glow(f, (W / 2, 640), TEAMS[wv][2], 1.4 * min(1, sc), 220)
                blit(f, *emoji_scaled(self.BADGE[wv], max(4, d)), int(W / 2 - d / 2), int(650 - d / 2))
                if te > 0.6:
                    cs = ease_out_back(min(1.0, (te - 0.6) / 0.35)); cd = int(150 * max(0.05, cs))
                    blit(f, *emoji_scaled(self.CROWN, cd), int(W / 2 - cd / 2), int(650 - d / 2 - cd * 0.75 + 8 * math.sin(te * 4)))
                put_text(f, f"{TEAMS[wv][1]} {champ_word(TEAMS[wv][1])}!", 76, W / 2, 920, TEAMS[wv][2], stroke=7, stroke_col=(255, 255, 255), scale=max(0.05, sc), maxw=680)
                if te > 1.0:
                    put_text(f, "Seu país foi longe?", 40, W / 2, 1020, (255, 255, 255), stroke=3, stroke_col=(30, 20, 30), alpha=min(1, (te - 1.0) / 0.3))
                    put_text(f, "Comenta aqui", 34, W / 2 - 24, 1080, (200, 220, 255), stroke=3, stroke_col=(30, 20, 30), alpha=min(1, (te - 1.0) / 0.3))
                    blit(f, *emoji_scaled(self.FINGER, 44), int(W / 2 + 100), int(1058 + 5 * math.sin(te * 6)))
            img = np.clip(f, 0, 255).astype(np.uint8)
            shake = max(0.0, 10 * (1 - te / 0.5))
        else:
            shake = 0.0
            for (t0, tm, rest) in elims:
                if 0 <= t - t0 < 0.4:
                    shake = max(shake, 9 * (1 - (t - t0) / 0.4))
        if shake > 0.5:
            M = np.float32([[1, 0, srng.normal() * shake], [0, 1, srng.normal() * shake]])
            img = cv2.warpAffine(img, M, (W, H))
        return img


# ======================= áudio =======================
SR = 44100


def make_audio(events, dur, path, seed=2):
    rng = np.random.default_rng(seed)
    N = int(SR * dur); out = np.zeros((N, 2), np.float32)
    def env(n, tau): return np.exp(-np.arange(n) / SR / tau)
    def lp(x, a): return lfilter([a], [1, a - 1], x)
    def tt(n): return np.arange(n) / SR
    def tone(f, du, tau, harm=((1, 1.0),)):
        n = int(SR * du); t = tt(n)
        return sum(a * np.sin(2 * np.pi * f * h * t) for h, a in harm) * env(n, tau) * np.minimum(1, t / 0.002)
    scale = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5, 1174.7, 1318.5]
    def note(i):
        return tone(scale[i], 0.35, 0.1, ((1, 1), (2, 0.4), (3, 0.15)))
    def click():
        n = int(SR * 0.06); t = tt(n)
        return np.sin(2 * np.pi * 2600 * t) * env(n, 0.005)
    def press():
        n = int(SR * 0.25); t = tt(n)
        return np.sin(2 * np.pi * (140 - 60 * t / 0.25) * t) * env(n, 0.05) + (np.sin(2 * np.pi * 760 * t)) * env(n, 0.02) * 0.4
    def clack():
        n = int(SR * 0.1); t = tt(n)
        return (np.sin(2 * np.pi * 1200 * t) + 0.6 * np.sin(2 * np.pi * 2200 * t)) * env(n, 0.015)
    def steal():
        n = int(SR * 0.3); t = tt(n)
        return np.sin(2 * np.pi * np.cumsum(1800 * np.exp(-t * 9) + 200) / SR) * env(n, 0.1)
    def beep():
        n = int(SR * 0.16); t = tt(n)
        return np.sign(np.sin(2 * np.pi * 1760 * t)) * 0.4 * np.minimum(1, (0.16 - t) / 0.01) * np.minimum(1, t / 0.005)
    def elim():
        n = int(SR * 0.9); t = tt(n)
        bz = (np.sign(np.sin(2 * np.pi * 110 * t)) + 0.6 * np.sign(np.sin(2 * np.pi * 116 * t)))
        bz = lp(bz, 0.25) * np.minimum(1, (0.6 - t) / 0.05).clip(0, 1)
        nz = rng.normal(0, 1, n)
        burst = lp(nz, 0.3) * env(n, 0.12) * 1.5
        return bz + burst
    def final():
        s = np.zeros(int(SR * 1.6))
        for f in (392.0, 493.9, 587.3, 784.0):
            n = int(SR * 1.4); t = tt(n)
            x = lp(np.sign(np.sin(2 * np.pi * f * t)) * 0.3 + np.sin(2 * np.pi * f * t), 0.25) * np.minimum(1, t / 0.03) * env(n, 0.7)
            s[:n] += x
        return s
    def gong():
        n = int(SR * 1.5); t = tt(n)
        return sum(a * np.sin(2 * np.pi * 1250 * r * t) * env(n, 0.35 * dd) for r, a, dd in ((1, 1, 1), (2.4, 0.5, 0.7), (3.9, 0.3, 0.5)))
    def boom():
        n = int(SR * 2.0); t = tt(n); nz = rng.normal(0, 1, n)
        return lp(nz, 0.02) * 6 * env(n, 0.4) + np.sin(2 * np.pi * np.cumsum(70 - 40 * np.minimum(t, 1)) / SR) * env(n, 0.45)
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
        n = int(SR * 3.5); t = tt(n)
        nz = lfilter([0.3], [1, -0.7], rng.normal(0, 1, n))
        e = np.minimum(1, t / 0.4) * np.exp(-np.maximum(0, t - 1.6) / 0.9)
        claps = np.zeros(n)
        for _ in range(170):
            i0 = int(rng.uniform(0, 2.8) * SR); m = 300
            if i0 + m < n:
                claps[i0:i0 + m] += rng.normal(0, 1, m) * env(m, 0.002) * rng.uniform(0.3, 1)
        return (nz * 2 + claps) * e
    base = {"tick": click(), "press": press(), "clack": clack(), "steal": steal(), "beep": beep(), "elim": elim(),
            "final": final(), "gong": gong(), "boom": boom(), "fanfare": fanfare(), "crowd": crowd()}
    for i in range(NT):
        base["n%d" % i] = note(i)
    base = {k: v / np.abs(v).max() for k, v in base.items()}
    gain = {"elim": 1.5, "final": 1.3, "fanfare": 1.9, "crowd": 1.2, "boom": 1.4, "beep": 0.9, "gong": 1.2}
    for (t, kind, vol) in events:
        s = base[kind] * vol * gain.get(kind, 1.0) * (0.55 if kind.startswith("n") else 1.0)
        i0 = int(t * SR)
        if i0 >= N:
            continue
        seg = s[: N - i0]
        pan = ((int(kind[1:]) / (NT - 1)) - 0.5) * 0.8 if kind.startswith("n") else 0.0
        out[i0:i0 + len(seg), 0] += seg * (1 - pan); out[i0:i0 + len(seg), 1] += seg * (1 + pan)
    for d, g in ((0.05, 0.2), (0.12, 0.12), (0.21, 0.06)):
        k = int(d * SR); out[k:] += out[:-k].copy() * g
    out /= np.abs(out).max() / 0.9
    w = wave.open(path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes()); w.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    ar = A.Arena(CFG)
    od = "out"
    if mode == "info":
        print("C", ar.C, "r_in", ar.r_in, "RB", ar.RB, "R1", ar.R1, "R2", ar.R2, "D", ar.D, "Wb", ar.Wb)
    elif mode == "search":
        for seed in range(int(sys.argv[2]), int(sys.argv[3])):
            _, au, pops, elims, parts, st, early = simulate(ar, seed, keep=False)
            sc, desc = score(elims, st, early)
            print("seed", seed, "score", sc, desc, flush=True)
    else:
        seed = int(sys.argv[2])
        frames, audio, pops, elims, parts, st, early = simulate(ar, seed)
        print("duracao", len(frames) / FPS, score(elims, st, early)[1], flush=True)
        rd = Renderer(ar); srng = np.random.default_rng(1)
        conf = Conf(3, (st["t_end"] or 0) + 0.3, (TEAMS[st["champion"] or 0][2], (255, 255, 255)))
        if mode == "preview":
            want = set(int(x) for x in sys.argv[3:])
            for i, fr in enumerate(frames):
                im = rd.render(fr, pops, elims, parts, st, srng, conf)
                if i in want:
                    cv2.imwrite(f"{od}/p_{i:04d}.png", im)
        else:
            enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                    "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                    "-pix_fmt", "yuv420p", f"{od}/v_tmp.mp4"], stdin=subprocess.PIPE)
            for fr in frames:
                enc.stdin.write(rd.render(fr, pops, elims, parts, st, srng, conf).tobytes())
            enc.stdin.close(); enc.wait()
            dur = len(frames) / FPS
            make_audio(sorted(audio), dur, f"{od}/audio.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{od}/v_tmp.mp4", "-i", f"{od}/audio.wav", "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(dur), f"{od}/tmp.mp4"], check=True)
            os.replace(f"{od}/tmp.mp4", f"{od}/batalha_paises.mp4")
            print("ok", flush=True)
