"""DUELO COM ITENS — 2 bolões-bandeira quicando num círculo, barras de vida no topo.
A cada ~4 s aparece um ITEM num anel brilhante (espinhos, espada de brinquedo, escudo, ímã, raio).
Quem pega ganha o item ("BRAZIL GOT THE SWORD!"); com item, o choque tira vida do outro
(faíscas/confete na cor do time, nada de sangue) e solta bolinhas da cor do time -> arena lota.
Quem zera perde -> confete na cor do vencedor, arena explode, tela final bilíngue.
Uso: python3 duelo_itens.py preview usa_brasil 30 300 ... | render usa_brasil
Nunca usar pessoas reais / fotos: só bandeiras (emoji Noto), cores e itens."""
import sys, os, math, subprocess
import numpy as np
import cv2
from scipy.spatial import cKDTree
from bolas import W, H, FPS, put_text, blit, emoji, make_ball_sprite, make_audio, P, SH, ease_out_back

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = 3
DT = 1.0 / (FPS * SUB)
CX, CY, RA = 360.0, 700.0, 300.0
RBIG = 46.0
RB = 7.0
SPEED = 330.0

ITEMS = {  # nome, emoji, dano, nome_pt
    "SPIKES": ("\U0001F335", 0, "os espinhos"),
    "SWORD": ("\U0001F5E1️", 0, "a espada"),
    "SHIELD": ("\U0001F6E1️", 0, "o escudo"),
    "MAGNET": ("\U0001F9F2", 0, "o ímã"),
    "LIGHTNING": ("⚡", 0, "o raio"),
}

CONFIGS = {
    "usa_brasil": dict(
        teams=[dict(name="USA", pt="EUA", flag="\U0001F1FA\U0001F1F8", ring=(200, 80, 30), ring2=(50, 50, 220), cols=[(225, 95, 35), (60, 60, 230), (235, 235, 235)]),
               dict(name="BRAZIL", pt="Brasil", flag="\U0001F1E7\U0001F1F7", ring=(60, 170, 20), ring2=(30, 215, 250), cols=[(60, 175, 25), (30, 215, 250), (160, 90, 20)])],
        # (tempo, item, quem pega, dano)  -> vida disputada e virada no final
        script=[(2.5, "SPIKES", 0, 14), (6.5, "SWORD", 1, 16), (10.5, "LIGHTNING", 0, 18), (14.5, "SHIELD", 1, 0),
                (18.0, "SWORD", 0, 15), (21.5, "SPIKES", 0, 20), (25.0, "LIGHTNING", 0, 22), (28.5, "MAGNET", 1, 30),
                (32.0, "SWORD", 1, 26), (35.5, "LIGHTNING", 1, 30)],
        winner=1, cap_end=1600, seed=5),
}


def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    T = cfg["teams"]
    C = np.array([CX, CY])
    bp = np.array([[CX - 120, CY - 40], [CX + 120, CY + 40]], float)
    bv = np.array([[SPEED * 0.8, -SPEED * 0.6], [-SPEED * 0.7, SPEED * 0.71]])
    hp = [100.0, 100.0]; hp_show = [100.0, 100.0]
    hold = [None, None]; shield = [False, False]
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2)); team = np.zeros(0, np.int32); sub = np.zeros(0, np.int32)
    item = None              # dict(kind, p, t0, taker, dmg)
    script = list(cfg["script"])
    msgs, conf, audio, frames = [], [], [], []
    st = dict(over=None, t_expl=None, winner=None, final=None)
    last_hit = -9.0
    t_tick = 0.5; ti = 0; note = 0
    NF = int(52 * FPS)
    frags = []

    def spawn(n, p0, v0, tm):
        nonlocal pos, prev, team, sub
        if n <= 0:
            return
        ang = rng.uniform(0, 2 * math.pi, n)
        rr = RBIG + RB + rng.uniform(1, 14, n)
        p = p0 + np.stack([np.cos(ang), np.sin(ang)], 1) * rr[:, None]
        d = p - C; dn = np.linalg.norm(d, axis=1)
        o = dn > RA - RB - 1
        p[o] = C + d[o] / dn[o, None] * (RA - RB - 1)
        v = np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(120, 320, (n, 1)) + v0 * 0.3
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        team = np.concatenate([team, np.full(n, tm, np.int32)]); sub = np.concatenate([sub, rng.integers(0, 3, n).astype(np.int32)])

    def confetti(p, cols, n, sp=420):
        for _ in range(n):
            a = rng.uniform(0, 2 * math.pi); s = rng.uniform(0.3, 1.0) * sp
            conf.append(dict(p=np.array(p, float), v=np.array([math.cos(a) * s, math.sin(a) * s]), c=cols[rng.integers(0, len(cols))],
                             t0=t, life=rng.uniform(0.6, 1.1), rot=rng.uniform(0, 6), w=rng.uniform(-12, 12)))

    for fi in range(NF):
        t = fi / FPS
        over = st["over"] is not None
        while t_tick <= t:
            if st["t_expl"] is None:
                audio.append((t_tick, "tick" if ti % 2 == 0 else "tock", 0.16)); ti += 1
            t_tick += 0.5 if min(hp) > 30 else 0.25
        # ---- itens
        if not over and item is None and script and t >= script[0][0]:
            ts, kind, tk, dmg = script.pop(0)
            # aparece um pouco à frente de quem vai pegar
            fut = bp[tk] + bv[tk] * 0.7
            d = fut - C; dn = np.linalg.norm(d)
            if dn > RA - 70:
                fut = C + d / dn * (RA - 70)
            item = dict(kind=kind, p=fut, t0=t, taker=tk, dmg=dmg)
            audio.append((t, "clink", 0.5))
        if item is not None and item.get("got") is None:
            tk = item["taker"]
            if np.linalg.norm(bp[tk] - item["p"]) < RBIG + 26:
                item["got"] = t
                if item["kind"] == "SHIELD":
                    shield[tk] = True
                else:
                    hold[tk] = (item["kind"], item["dmg"])
                msgs.append((t, f"{T[tk]['name']} GOT THE {item['kind']}!", f"{T[tk]['pt']} pegou {ITEMS[item['kind']][2]}!", tk, item["kind"]))
                audio.append((t, "trim", 0.8)); audio.append((t, "plim", 0.5))
                if item["kind"] == "MAGNET":       # ímã: puxa as bolinhas do outro e elas mudam de cor
                    near = (team != tk) & (np.linalg.norm(pos - bp[tk], axis=1) < 190)
                    team[near] = tk
                item = None
        # ---- física dos bolões
        for s in range(SUB):
            ts_ = t + s * DT
            if st["t_expl"] is None:
                for k in range(2):
                    if over and k != st["winner"]:
                        continue
                    tgt = None; spd_k = SPEED
                    if item is not None and item["taker"] == k and item.get("got") is None:
                        tgt = item["p"]
                    if hold[k] is not None and not over:
                        tgt = bp[1 - k] + bv[1 - k] * 0.25; spd_k = SPEED * 1.35
                    if tgt is not None:
                        want = math.atan2(tgt[1] - bp[k][1], tgt[0] - bp[k][0]); cur = math.atan2(bv[k][1], bv[k][0])
                        da = (want - cur + math.pi) % (2 * math.pi) - math.pi
                        bv[k] = rot(bv[k], float(np.clip(da, -5.0 * DT, 5.0 * DT)))
                    sp = np.linalg.norm(bv[k]); bv[k] *= spd_k / max(sp, 1e-6)
                    bp[k] += bv[k] * DT
                    d = bp[k] - C; dn = np.linalg.norm(d)
                    if dn > RA - RBIG:
                        nrm = d / dn; bp[k] = C + nrm * (RA - RBIG)
                        vn = bv[k] @ nrm
                        if vn > 0:
                            bv[k] -= 2 * vn * nrm
                            bv[k] = rot(bv[k], rng.normal(0, 0.12))
                            # cada batida na parede solta bolinhas (enche a arena)
                            if not over:
                                target = cfg["cap_end"] * math.exp(0.105 * (ts_ - 37.0))
                                n = int(np.clip((target - len(pos)) * 0.6, 1, 70)) if t > 1.0 else 0
                                spawn(n, bp[k], bv[k], k)
                # bolão x bolão
                if not over:
                    d = bp[1] - bp[0]; dn = np.linalg.norm(d)
                    if dn < 2 * RBIG:
                        nrm = d / max(dn, 1e-6)
                        ov = 2 * RBIG - dn
                        bp[0] -= nrm * ov / 2; bp[1] += nrm * ov / 2
                        v0n, v1n = bv[0] @ nrm, bv[1] @ nrm
                        if v0n - v1n > 0:
                            bv[0] += (v1n - v0n) * nrm; bv[1] += (v0n - v1n) * nrm
                        if ts_ - last_hit > 0.35:
                            last_hit = ts_
                            hitp = (bp[0] + bp[1]) / 2
                            audio.append((ts_, "clink", 0.35))
                            for a in (0, 1):
                                if hold[a] is None:
                                    continue
                                kind, dmg = hold[a]; b = 1 - a
                                hold[a] = None
                                if shield[b]:
                                    shield[b] = False
                                    msgs.append((ts_, "BLOCKED!", "Bloqueou!", b, "SHIELD"))
                                    confetti(hitp, [(240, 240, 240), (200, 200, 200)], 30, 300)
                                    audio.append((ts_, "trim", 0.7))
                                    continue
                                hp[b] = max(0.0, hp[b] - dmg)
                                confetti(bp[b], T[a]["cols"][:2], 60, 520)
                                spawn(int(14 + dmg * 1.6), bp[a], bv[a], a)
                                msgs.append((ts_, f"-{dmg}", "", b, "DMG"))
                                audio.append((ts_, "plim", 0.9)); audio.append((ts_, "tom%d" % (note % 8), 0.7)); note += 1
                                if hp[b] < 35:
                                    audio.append((ts_, "rumble", 0.5))
                                if hp[b] <= 0 and st["over"] is None:
                                    st.update(over=ts_, winner=a)
                                    confetti(bp[b], T[a]["cols"][:2], 160, 700)
                                    audio.append((ts_, "riser", 0.9))
            # ---- bolinhas
            if len(pos):
                v = (pos - prev) / DT * 0.999
                g = np.array([0, 900.0]) if st["t_expl"] is not None else np.zeros(2)
                spd = np.linalg.norm(v, axis=1, keepdims=True)
                v = np.where(spd > 700, v / np.maximum(spd, 1e-6) * 700, v) if st["t_expl"] is None else v
                prev = pos; pos = pos + v * DT + g * DT * DT
                if st["t_expl"] is None:
                    for it in range(2):
                        pr = cKDTree(pos).query_pairs(2 * RB, output_type="ndarray")
                        if len(pr):
                            i, j = pr[:, 0], pr[:, 1]
                            d = pos[j] - pos[i]; dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4)
                            corr = d * ((2 * RB - dist) / dist * 0.5 * 0.85)[:, None]
                            dp = np.zeros_like(pos); np.add.at(dp, i, -corr); np.add.at(dp, j, corr); pos += dp
                        for k in range(2):
                            if over and k != st["winner"]:
                                continue
                            d = pos - bp[k]; dn = np.linalg.norm(d, axis=1)
                            o = dn < RBIG + RB
                            if o.any():
                                nrm = d[o] / np.maximum(dn[o], 1e-4)[:, None]
                                pos[o] = bp[k] + nrm * (RBIG + RB)
                                vv = (pos[o] - prev[o]) / DT
                                vn = (vv * nrm).sum(1, keepdims=True)
                                vv += nrm * np.maximum(0, (bv[k] * nrm).sum(1, keepdims=True) * 1.3 - vn)
                                prev[o] = pos[o] - vv * DT
                        d = pos - C; dn = np.linalg.norm(d, axis=1)
                        o = dn > RA - RB
                        if o.any():
                            nrm = d[o] / dn[o, None]
                            pos[o] = C + nrm * (RA - RB)
                            vv = pos[o] - prev[o]; vn = (vv * nrm).sum(1, keepdims=True)
                            prev[o] = pos[o] - (vv - np.maximum(vn, 0) * nrm * 1.8)
        # ---- fim: confete do vencedor e a arena explode
        if over and st["t_expl"] is None and t >= st["over"] + 1.6:
            st["t_expl"] = t; st["final"] = len(pos)
            audio.append((t, "boom", 1.0)); audio.append((t, "sparkle", 0.9))
            d = pos - C; dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
            sp = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / RA)
            prev = pos - (d / dist * sp + rng.normal(0, 120, pos.shape)) * DT
            for k in range(40):
                a0 = 2 * math.pi * k / 40; a1 = 2 * math.pi * (k + 1) / 40
                pa = C + RA * np.array([math.cos(a0), math.sin(a0)]); pb = C + RA * np.array([math.cos(a1), math.sin(a1)])
                mm = (pa + pb) / 2; dv = (mm - C) / RA
                frags.append(dict(a=pa - mm, b=pb - mm, c=mm, rot=0.0, v=dv * rng.uniform(500, 1100), w=rng.uniform(-6, 6)))
            confetti(bp[st["winner"]], T[st["winner"]]["cols"][:2], 200, 900)
        if st["t_expl"] is not None:
            kp = (pos[:, 1] < H + 40) & (pos[:, 0] > -40) & (pos[:, 0] < W + 40)
            pos, prev, team, sub = pos[kp], prev[kp], team[kp], sub[kp]
            for fg in frags:
                fg["c"] = fg["c"] + fg["v"] / FPS; fg["v"] = fg["v"] + np.array([0, 900.0]) / FPS; fg["rot"] += fg["w"] / FPS
        for c in conf:
            c["p"] = c["p"] + c["v"] / FPS; c["v"] = c["v"] * 0.95 + np.array([0, 500.0]) / FPS; c["rot"] += c["w"] / FPS
        conf[:] = [c for c in conf if t - c["t0"] < c["life"]]
        for k in range(2):
            hp_show[k] += (hp[k] - hp_show[k]) * 0.18
        frames.append(dict(t=t, bp=bp.copy(), pos=pos.copy(), team=team.copy(), sub=sub.copy(), hp=list(hp_show), hpi=list(hp),
                           hold=list(hold), shield=list(shield), item=dict(item) if item else None, conf=[dict(c) for c in conf],
                           frags=[dict(f) for f in frags], n=len(pos)))
        if st["t_expl"] is not None and t > st["t_expl"] + 4.2:
            break
    return frames, audio, msgs, st


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Rend:
    def __init__(self, cfg):
        self.cfg = cfg; T = cfg["teams"]
        self.SPR = [[make_ball_sprite(c, RB) for c in tm["cols"]] for tm in T]
        self.SS = self.SPR[0][0][0].shape[0]; self.HALF = self.SS // 2
        bg = np.zeros((H, W, 3), np.float32); bg[:] = (34, 18, 10)
        bg[::40, :] += 8; bg[:, ::40] += 8
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.clip(1 - np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2) / 800, 0, 1) ** 1.5
        bg += g[..., None] * np.array((30, 16, 8), np.float32)
        self.BG = bg
        # bolão com bandeira
        self.BIG = []
        for tm in T:
            s = int(RBIG * 2 + 12); im = np.zeros((s, s, 3), np.float32); al = np.zeros((s, s), np.float32)
            c = (s // 2, s // 2)
            cv2.circle(im, c, int(RBIG), tm["ring"], -1, cv2.LINE_AA); cv2.circle(al, c, int(RBIG), 1.0, -1, cv2.LINE_AA)
            cv2.circle(im, c, int(RBIG - 7), tm["ring2"], -1, cv2.LINE_AA)
            cv2.circle(im, c, int(RBIG - 12), (245, 245, 245), -1, cv2.LINE_AA)
            pre = im * al[..., None]; inv = (1 - al)[..., None]
            self.BIG.append((pre, inv, s))

    def render(self, fr, msgs, st):
        T = self.cfg["teams"]; t = fr["t"]
        f = self.BG.copy()
        ex = st["t_expl"] is not None and t >= st["t_expl"]
        # bolinhas
        ip = np.round(fr["pos"]).astype(np.int32)
        for k in range(len(ip)):
            pre, inv = self.SPR[fr["team"][k]][fr["sub"][k]]
            x0, y0 = ip[k, 0] - self.HALF, ip[k, 1] - self.HALF
            if 0 <= x0 and 0 <= y0 and x0 + self.SS <= W and y0 + self.SS <= H:
                reg = f[y0:y0 + self.SS, x0:x0 + self.SS]; reg *= inv; reg += pre
            else:
                blit(f, pre, inv, x0, y0)
        img = np.clip(f, 0, 255).astype(np.uint8)
        low = min(fr["hpi"])
        wob = 0.0 if low > 35 else (35 - low) / 35 * 6
        if not ex:
            pts = []
            for a in np.linspace(0, 2 * math.pi, 240, endpoint=False):
                r = RA + wob * math.sin(a * 7 + t * 20)
                pts.append((CX + r * math.cos(a), CY + r * math.sin(a)))
            cv2.polylines(img, [(np.array(pts) * SH).astype(np.int32)], True, (245, 245, 245), 3, cv2.LINE_AA, 4)
        for fg in fr["frags"]:
            cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"]); R = np.array([[cr, -sr], [sr, cr]])
            cv2.line(img, P(fg["c"] + R @ fg["a"]), P(fg["c"] + R @ fg["b"]), (245, 245, 245), 4, cv2.LINE_AA, 4)
        # item no anel brilhante
        it = fr["item"]
        if it is not None and not ex:
            age = t - it["t0"]; sc = ease_out_back(min(1.0, age / 0.35))
            r = int(30 * max(0.1, sc))
            gl = np.zeros_like(img)
            cv2.circle(gl, P(it["p"]), int((r + 6) * SH), (80, 220, 255), 10, cv2.LINE_AA, 4)
            img = cv2.add(img, (cv2.GaussianBlur(gl, (0, 0), 7) * (0.8 + 0.2 * math.sin(t * 6))).astype(np.uint8))
            cv2.circle(img, P(it["p"]), int((r + 4) * SH), (120, 235, 255), 3, cv2.LINE_AA, 4)
        # confete
        for c in fr["conf"]:
            a = 1 - (t - c["t0"]) / c["life"]
            if a <= 0:
                continue
            u = np.array([math.cos(c["rot"]), math.sin(c["rot"])]) * 5
            cv2.line(img, P(c["p"] - u), P(c["p"] + u), c["c"], 5, cv2.LINE_AA, 4)
        f = img.astype(np.float32)
        if it is not None and not ex:
            age = t - it["t0"]; sc = ease_out_back(min(1.0, age / 0.35))
            d = int(40 * max(0.1, sc))
            blit(f, *emoji(ITEMS[it["kind"]][0], d), int(it["p"][0] - d / 2), int(it["p"][1] - d / 2))
        # bolões
        for k in range(2):
            if st["over"] is not None and k != st["winner"] and t >= st["over"]:
                continue
            if ex:
                continue
            pre, inv, s = self.BIG[k]
            b = fr["bp"][k]
            if fr["shield"][k]:
                cv2.circle(f, P(b), int((RBIG + 9) * SH), (255, 230, 150), 4, cv2.LINE_AA, 4)
            hd = fr["hold"][k]
            if hd is not None and hd[0] == "SPIKES":
                for j in range(12):
                    a = t * 3 + j * math.pi / 6
                    u = np.array([math.cos(a), math.sin(a)]); n_ = np.array([-u[1], u[0]])
                    tri = np.array([b + u * (RBIG - 2) + n_ * 8, b + u * (RBIG + 18), b + u * (RBIG - 2) - n_ * 8])
                    cv2.fillPoly(f, [(tri * SH).astype(np.int32)], (200, 205, 215), cv2.LINE_AA, 4)
            blit(f, pre, inv, int(b[0] - s / 2), int(b[1] - s / 2))
            blit(f, *emoji(T[k]["flag"], 58), int(b[0] - 29), int(b[1] - 29))
            if hd is not None and hd[0] != "SPIKES":
                blit(f, *emoji(ITEMS[hd[0]][0], 38), int(b[0] + RBIG * 0.55), int(b[1] - RBIG * 1.25))
            if fr["shield"][k]:
                blit(f, *emoji(ITEMS["SHIELD"][0], 34), int(b[0] - RBIG * 1.25), int(b[1] - RBIG * 1.2))
        # HUD: título + barras de vida
        if not ex:
            put_text(f, f"{T[0]['name']} vs {T[1]['name']}", 52, W / 2, 70, (255, 255, 255), stroke=5, stroke_col=(30, 20, 40))
            put_text(f, f"{T[0]['pt']} contra {T[1]['pt']}", 24, W / 2, 112, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40))
            for k in range(2):
                x0 = 70 if k == 0 else 382; wbar = 268; y0 = 158
                cv2.rectangle(f, (x0, y0), (x0 + wbar, y0 + 30), (40, 30, 30), -1)
                fill = int(wbar * fr["hp"][k] / 100)
                col = T[k]["ring"]
                if k == 0:
                    cv2.rectangle(f, (x0, y0), (x0 + fill, y0 + 30), col, -1)
                else:
                    cv2.rectangle(f, (x0 + wbar - fill, y0), (x0 + wbar, y0 + 30), col, -1)
                cv2.rectangle(f, (x0, y0), (x0 + wbar, y0 + 30), (240, 240, 240), 2)
                put_text(f, f"{int(round(fr['hpi'][k]))}", 24, x0 + wbar / 2, y0 + 15, (255, 255, 255), stroke=3, stroke_col=(20, 20, 20))
                fx = 18 if k == 0 else W - 64
                blit(f, *emoji(T[k]["flag"], 46), fx, y0 - 8)
            put_text(f, "HP", 22, W / 2, 173, (220, 220, 220), stroke=2, stroke_col=(20, 20, 20))
            # avisos
            for (t0, en, pt, k, kind) in msgs:
                dt_ = t - t0
                if kind == "DMG":
                    if 0 <= dt_ < 0.9:
                        b = fr["bp"][k]
                        put_text(f, en, 40, b[0], b[1] - 70 - dt_ * 50, T[1 - k]["cols"][1], stroke=4, stroke_col=(255, 255, 255), alpha=min(1, (0.9 - dt_) * 3))
                    continue
                if 0 <= dt_ < 1.8 and not any(0 <= t - m[0] < 1.8 and m[0] > t0 and m[4] != "DMG" for m in msgs):
                    sc = ease_out_back(min(1.0, dt_ / 0.3)); a = min(1.0, (1.8 - dt_) * 3)
                    put_text(f, en, 40, W / 2, 262, T[k]["ring2"] if k == 1 else (255, 255, 255), stroke=5, stroke_col=T[k]["ring"], scale=max(0.05, sc), alpha=a)
                    put_text(f, pt, 24, W / 2, 302, (215, 220, 230), stroke=3, stroke_col=(30, 20, 40), scale=max(0.05, sc), alpha=a)
            if st["over"] is not None and t >= st["over"]:
                put_text(f, "K.O.!", 70, W / 2, CY, (255, 255, 255), stroke=6, stroke_col=T[st["winner"]]["ring"], scale=ease_out_back(min(1.0, (t - st["over"]) / 0.3)))
            else:
                put_text(f, "Follow for more!", 30, W / 2, CY + RA + 62, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40))
                put_text(f, "Segue a página pra ver mais", 22, W / 2, CY + RA + 96, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40))
        img = np.clip(f, 0, 255).astype(np.uint8)
        if ex:
            te = t - st["t_expl"]; wk = st["winner"]
            fl = max(0.0, 0.7 * (1 - te / 0.4))
            if fl > 0:
                img = cv2.addWeighted(img, 1 - fl, np.full_like(img, 255), fl, 0)
            if te > 0.5:
                k = min(1.0, (te - 0.5) / 0.4)
                img = (img.astype(np.float32) * (1 - 0.55 * k)).astype(np.uint8)
                f = img.astype(np.float32)
                sc = ease_out_back(min(1.0, (te - 0.6) / 0.4)) if te > 0.6 else 0.05
                d = int(170 * max(0.1, sc))
                blit(f, *emoji(T[wk]["flag"], d), int(W / 2 - d / 2), int(470 - d / 2))
                put_text(f, f"{T[wk]['name']} WINS!", 76, W / 2, 640, T[wk]["ring2"], stroke=7, stroke_col=T[wk]["ring"], scale=max(0.05, sc))
                put_text(f, f"{T[wk]['pt']} venceu!", 32, W / 2, 702, (215, 220, 230), stroke=3, stroke_col=(30, 20, 40), scale=max(0.05, sc))
                if te > 1.2:
                    a = min(1, (te - 1.2) / 0.3)
                    put_text(f, "Who were you rooting for?", 40, W / 2, 790, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40), alpha=a)
                    put_text(f, "Pra quem você torceu?", 24, W / 2, 830, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40), alpha=a)
                    blit(f, *emoji("\U0001F447", 40), int(W / 2 - 20), int(855 + 4 * math.sin(t * 6)))
                if te > 1.7:
                    a = min(1, (te - 1.7) / 0.3)
                    put_text(f, "Follow for more!", 40, W / 2, 945, (200, 230, 255), stroke=3, stroke_col=(30, 20, 40), alpha=a)
                    put_text(f, "Segue a página pra ver mais", 24, W / 2, 985, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40), alpha=a)
                img = np.clip(f, 0, 255).astype(np.uint8)
        return img


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    cfg = CONFIGS[name]
    od = os.path.join(HERE, f"out_{name}"); os.makedirs(od, exist_ok=True)
    frames, audio, msgs, st = simulate(cfg)
    print(name, "duracao", len(frames) / FPS, "KO", st["over"], "explode", st["t_expl"], "vencedor", st["winner"], "bolinhas", st["final"],
          "msgs", [(round(m[0], 1), m[1]) for m in msgs], flush=True)
    rd = Rend(cfg)
    if mode == "sim":
        sys.exit()
    if mode == "preview":
        want = set(int(x) for x in sys.argv[3:])
        for i, fr in enumerate(frames):
            if i in want:
                cv2.imwrite(f"{od}/p_{i:04d}.png", rd.render(fr, msgs, st))
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", f"{od}/v_tmp.mp4"], stdin=subprocess.PIPE)
        for fr in frames:
            enc.stdin.write(rd.render(fr, msgs, st).tobytes())
        enc.stdin.close(); enc.wait()
        dur = len(frames) / FPS
        make_audio(sorted(audio, key=lambda e: e[0]), dur, f"{od}/audio.wav", "maior")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{od}/v_tmp.mp4", "-i", f"{od}/audio.wav", "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(dur), f"{od}/tmp.mp4"], check=True)
        os.replace(f"{od}/tmp.mp4", f"{od}/{name}.mp4")
        print("ok", f"{od}/{name}.mp4", flush=True)
