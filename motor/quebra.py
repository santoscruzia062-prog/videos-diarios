"""QUEBRA-QUADRADOS: 1 bolinha contra dezenas de quadrados numerados.
Cada batida tira 1 ponto do quadrado; quadrado quebrado vira bolinha nova (multiplica).
Quando acaba tudo: 3 rodadas (bolinhas dobram, borda treme cada vez mais) e EXPLODE.

Uso:  python3 quebra.py tune <nome>          -> acha o fator de dureza (limpa ~32 s)
      python3 quebra.py preview <nome> <k> f1 f2 ...
      python3 quebra.py render <nome> <k>
"""
import sys, os, math, subprocess
import numpy as np
import cv2
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bolas
from bolas import W, H, FPS, Arena, put_text, blit, emoji, make_ball_sprite, make_audio, PALETTES, P, SH, ease_out_back

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = 4
DT = 1.0 / (FPS * SUB)
R = 9.0

CONFIGS = {
    "quebra_quadrados": dict(shape="rounded", cy=660, palette="vivo", bg=(45, 25, 20), V=560.0, cell=44, rows=10,
                             scale="maior", seed=4),
    "quebra_circulo": dict(shape="circle", cy=660, palette="neon", bg=(50, 20, 45), V=560.0, cell=42, rows=9,
                           scale="menor", seed=7),
}


def build(cfg, k):
    rng = np.random.default_rng(cfg.get("seed", 1))
    ar = Arena(cfg["shape"], cfg["cy"])
    cell = cfg["cell"]; bs = cell - 4
    ncol = int((W - 100) // cell)
    x0 = W / 2 - ncol * cell / 2
    y0 = ar.top + 16
    bricks = []
    for r in range(cfg["rows"]):
        for c in range(ncol):
            bx, by = x0 + c * cell + 2, y0 + r * cell + 2
            corners = np.array([[bx, by], [bx + bs, by], [bx, by + bs], [bx + bs, by + bs]])
            sd, _, _ = ar.sample(corners, 1.0)
            if (sd > R * 2.4).all():
                base = 1 + (cfg["rows"] - 1 - r) * 0.55 + rng.uniform(0, 2.5)
                hpv = int(rng.integers(1, 3)) if r >= cfg["rows"] - 2 else max(1, int(round(base * k)))
                bricks.append((bx, by, hpv, r, c))
    B = np.array([[b[0], b[1]] for b in bricks], float)
    hp = np.array([b[2] for b in bricks], int)
    rowi = np.array([b[3] for b in bricks], int)
    grid = -np.ones((cfg["rows"] + 2, ncol + 2), int)
    for i, b in enumerate(bricks):
        grid[b[3] + 1, b[4] + 1] = i
    return ar, B, hp, rowi, grid, x0, y0, cell, bs


def simulate(cfg, k, keep=True):
    rng = np.random.default_rng(11)
    ar, B, hp, rowi, grid, gx0, gy0, cell, bs = build(cfg, k)
    hp0 = hp.copy()
    pal = PALETTES[cfg["palette"]]
    ncolor = len(pal)
    bcol = (rowi * 2 + (np.arange(len(B)) % 2)) % ncolor
    alive = np.ones(len(B), bool)
    V = cfg["V"]
    C = ar.C
    pos = np.array([[W / 2, ar.bottom - 60.0]]); a0 = -math.pi / 2 + 0.35
    vel = np.array([[math.cos(a0), math.sin(a0)]]) * V
    col = np.array([0])
    frames, audio, frags = [], [], []
    hitt = np.full(len(B), -9.0)
    st = dict(t_clear=None, rounds=0, lvl=0, s=1.0, t_lastpop=None, exploded=False, t_expl=None, final=None)
    NF = int(52 * FPS)
    t_tick, tick_i = 0.5, 0
    last_note, last_hit = -1.0, -1.0
    LV = [1.0, 1.02, 1.04, 1.065]

    for fi in range(NF):
        t = fi / FPS
        ex = st["exploded"]
        while t_tick <= t:
            if not ex:
                audio.append((t_tick, "tick" if tick_i % 2 == 0 else "tock", 0.16)); tick_i += 1
            fast = st["t_clear"] is not None and not ex
            t_tick += (0.5 / (1 + st["lvl"] * 1.2)) if fast else 0.5
        # rodadas finais
        if st["t_clear"] is not None and not ex:
            tc = st["t_clear"]
            for j, tr in enumerate((tc + 1.2, tc + 2.4, tc + 3.6)):
                if st["rounds"] == j and t >= tr:
                    st["rounds"] += 1; st["lvl"] = j + 1; st["t_lastpop"] = t
                    n = len(pos)
                    ang = rng.uniform(0, 2 * math.pi, n)
                    dv = np.stack([np.cos(ang), np.sin(ang)], 1)
                    pos = np.vstack([pos, pos + dv * R * 1.2]); vel = np.vstack([vel, dv * V])
                    col = np.concatenate([col, (col + 1 + j) % ncolor])
                    audio.append((t, "trim", 0.6)); audio.append((t, "rumble", 0.4 + 0.15 * j)); audio.append((t, "plim", 0.5))
            st["s"] += (LV[st["lvl"]] - st["s"]) * 0.08
            if st["rounds"] == 3 and t >= tc + 4.8:
                st.update(exploded=True, t_expl=t, final=len(pos)); ex = True
                audio.append((t - 1.0, "riser", 0.8)); audio.append((t, "boom", 1.0)); audio.append((t, "sparkle", 0.9))
                d = pos - C; dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                vel = d / dist * rng.uniform(700, 1500, (len(pos), 1)) + rng.normal(0, 120, pos.shape)
                Pp = ar.dense(st["s"]); step = max(1, len(Pp) // 40)
                for i in range(0, len(Pp), step):
                    a, b = Pp[i], Pp[(i + step) % len(Pp)]
                    mm = (a + b) / 2; dv = (mm - C) / (np.linalg.norm(mm - C) + 1e-6)
                    frags.append(dict(kind="line", a=a - mm, b=(b - mm) * 0.9, c=mm, rot=0.0, v=dv * rng.uniform(500, 1100),
                                      w=rng.uniform(-6, 6), t0=t))
        nhit, nbrk, brk_pts = 0, 0, []
        for s_ in range(SUB):
            if ex:
                vel = vel + np.array([0, 900.0]) * DT
                pos = pos + vel * DT
                continue
            pos = pos + vel * DT
            # bolinha x bolinha
            pr = cKDTree(pos).query_pairs(2 * R, output_type="ndarray") if len(pos) > 1 else np.zeros((0, 2), int)
            if len(pr):
                i, j = pr[:, 0], pr[:, 1]
                d = pos[j] - pos[i]; dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4); n = d / dist[:, None]
                corr = n * ((2 * R - dist) * 0.5)[:, None]
                dp = np.zeros_like(pos); np.add.at(dp, i, -corr); np.add.at(dp, j, corr); pos += dp
                rv = ((vel[j] - vel[i]) * n).sum(1)
                m = rv < 0
                if m.any():
                    imp = n[m] * rv[m][:, None]
                    dv = np.zeros_like(vel); np.add.at(dv, i[m], imp); np.add.at(dv, j[m], -imp); vel += dv
            # bolinha x quadrado
            if alive.any():
                ci = np.floor((pos[:, 0] - gx0) / cell).astype(int) + 1
                ri = np.floor((pos[:, 1] - gy0) / cell).astype(int) + 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        rr = ri + dy; cc = ci + dx
                        ok = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
                        bi = np.full(len(pos), -1); bi[ok] = grid[rr[ok], cc[ok]]
                        m = bi >= 0
                        if not m.any():
                            continue
                        m[m] = alive[bi[m]]
                        if not m.any():
                            continue
                        idx = np.where(m)[0]; b = bi[idx]
                        bx0, by0 = B[b, 0], B[b, 1]
                        cx = np.clip(pos[idx, 0], bx0, bx0 + bs); cy = np.clip(pos[idx, 1], by0, by0 + bs)
                        d = pos[idx] - np.stack([cx, cy], 1); dist = np.linalg.norm(d, axis=1)
                        h = dist < R
                        if not h.any():
                            continue
                        idx, b, d, dist = idx[h], b[h], d[h], dist[h]
                        inside = dist < 1e-3
                        n = np.where(inside[:, None], -vel[idx] / (np.linalg.norm(vel[idx], axis=1, keepdims=True) + 1e-6),
                                     d / np.maximum(dist, 1e-6)[:, None])
                        vn = (vel[idx] * n).sum(1)
                        app = vn < 0
                        vel[idx[app]] -= 2 * vn[app][:, None] * n[app]
                        cp = pos[idx] - d
                        pos[idx] = cp + n * R
                        bb = b[app]
                        if len(bb):
                            np.subtract.at(hp, bb, 1); hitt[bb] = t; nhit += len(bb)
                            dead = np.unique(bb[hp[bb] <= 0])
                            dead = dead[alive[dead]]
                            if len(dead):
                                alive[dead] = False; nbrk += len(dead)
                                ctr = B[dead] + bs / 2
                                ang = rng.uniform(0, 2 * math.pi, len(dead))
                                pos = np.vstack([pos, ctr]); vel = np.vstack([vel, np.stack([np.cos(ang), np.sin(ang)], 1) * V])
                                col = np.concatenate([col, bcol[dead]])
                                for q, cc_ in zip(dead, ctr):
                                    brk_pts.append(cc_)
                                    for f_ in range(5):
                                        a = rng.uniform(0, 2 * math.pi)
                                        frags.append(dict(kind="sq", c=cc_ + rng.normal(0, 6, 2), v=np.array([math.cos(a), math.sin(a)]) * rng.uniform(120, 320) + [0, -120],
                                                          rot=rng.uniform(0, 3), w=rng.uniform(-8, 8), col=pal[bcol[q]], sz=rng.uniform(5, 11), t0=t))
                                ci = np.floor((pos[:, 0] - gx0) / cell).astype(int) + 1
                                ri = np.floor((pos[:, 1] - gy0) / cell).astype(int) + 1
            # borda
            sd, gxx, gyy = ar.sample(pos, st["s"])
            o = sd < R
            if o.any():
                g = np.stack([gxx[o], gyy[o]], 1)
                pos[o] += g * (R - sd[o])[:, None]
                vn = (vel[o] * g).sum(1)
                vel[o] -= 2 * np.minimum(vn, 0)[:, None] * g
            # velocidade constante
            sp = np.linalg.norm(vel, axis=1, keepdims=True)
            vel = vel / np.maximum(sp, 1e-6) * V
        if not ex and st["t_clear"] is None and not alive.any():
            st["t_clear"] = t; audio.append((t, "plim", 0.9)); audio.append((t, "sparkle", 0.5))
        if nhit and t - last_hit > 0.05:
            audio.append((t, "clink", min(0.35, 0.1 + 0.03 * nhit))); last_hit = t
        if nbrk and t - last_note > 0.07:
            audio.append((t, "tom", 0.55)); last_note = t
        if ex:
            kp = (pos[:, 0] > -60) & (pos[:, 0] < W + 60) & (pos[:, 1] < H + 60) & (pos[:, 1] > -600)
            pos, vel, col = pos[kp], vel[kp], col[kp]
        for f_ in frags:
            f_["c"] = f_["c"] + f_["v"] / FPS; f_["v"] = f_["v"] + np.array([0, 900.0]) / FPS; f_["rot"] += f_["w"] / FPS
        frags = [f_ for f_ in frags if f_["kind"] == "line" or t - f_["t0"] < 0.8]
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), col=col.copy(), alive=alive.copy(), hp=hp.copy(), hitt=hitt.copy(),
                               s=st["s"], lvl=st["lvl"], t_clear=st["t_clear"], t_lastpop=st["t_lastpop"], exploded=ex,
                               t_expl=st["t_expl"], frags=[dict(f_) for f_ in frags], brk=brk_pts))
        if ex and t > st["t_expl"] + 4.0:
            break
        if not keep and t > 48 and st["t_clear"] is None:
            break
    return frames, audio, st, dict(ar=ar, B=B, hp0=hp0, bcol=bcol, bs=bs, pal=pal, nb=len(B))


class Rend:
    def __init__(self, cfg, info):
        self.cfg, self.info = cfg, info
        ar = info["ar"]; self.ar = ar
        self.pal = info["pal"]
        self.SPR = [make_ball_sprite(c, R) for c in self.pal]
        self.SS = self.SPR[0][0].shape[0]; self.HALF = self.SS // 2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.clip(1 - np.sqrt((xx - ar.C[0]) ** 2 + (yy - ar.C[1]) ** 2) / 760, 0, 1) ** 1.5
        bg = np.zeros((H, W, 3), np.float32); bg[:] = (8, 6, 8)
        self.BG = bg + g[..., None] * np.array(cfg.get("bg", (40, 22, 30)), np.float32)
        fl = (ar.mask > 0).astype(np.float32)
        self.BG = self.BG * (1 - fl[..., None] * 0.45) + fl[..., None] * 0.45 * np.array((26, 20, 24), np.float32)

    def render(self, fr, st, srng):
        info = self.info; ar = self.ar; pal = self.pal; t = fr["t"]; ex = fr["exploded"]
        img = self.BG.astype(np.uint8).copy() if not ex else (self.BG * 0.7).astype(np.uint8)
        B, bs = info["B"], info["bs"]
        # quadrados
        for i in np.where(fr["alive"])[0]:
            x, y = B[i]; c = pal[info["bcol"][i]]
            k = max(0.0, 1 - (t - fr["hitt"][i]) / 0.12)
            frac = fr["hp"][i] / max(1, info["hp0"][i])
            base = tuple(int(min(255, cc * (0.55 + 0.45 * frac) + k * 70)) for cc in c)
            x0, y0 = int(x), int(y); sz = int(bs)
            cv2.rectangle(img, (x0, y0), (x0 + sz, y0 + sz), base, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x0, y0), (x0 + sz, y0 + 5), tuple(min(255, cc + 60) for cc in base), -1)
            cv2.rectangle(img, (x0, y0), (x0 + sz, y0 + sz), tuple(int(cc * 0.5) for cc in base), 2, cv2.LINE_AA)
        f = img.astype(np.float32)
        for i in np.where(fr["alive"])[0]:
            x, y = B[i]
            put_text(f, str(int(fr["hp"][i])), 20, x + bs / 2, y + bs / 2 + 1, (255, 255, 255), stroke=3, stroke_col=(25, 20, 25))
        # bolinhas
        ip = np.round(fr["pos"]).astype(np.int32); cl = fr["col"]
        for k_ in range(len(ip)):
            pre, inv = self.SPR[cl[k_]]
            x0, y0 = ip[k_, 0] - self.HALF, ip[k_, 1] - self.HALF
            if 0 <= x0 and 0 <= y0 and x0 + self.SS <= W and y0 + self.SS <= H:
                reg = f[y0:y0 + self.SS, x0:x0 + self.SS]; reg *= inv; reg += pre
            else:
                blit(f, pre, inv, x0, y0)
        img = np.clip(f, 0, 255).astype(np.uint8)
        for fg in fr["frags"]:
            if fg["kind"] == "sq":
                al = max(0.0, 1 - (t - fg["t0"]) / 0.8)
                s2 = fg["sz"] * (0.5 + 0.5 * al); cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"])
                pts = np.array([[-s2, -s2], [s2, -s2], [s2, s2], [-s2, s2]]) @ np.array([[cr, sr], [-sr, cr]]) + fg["c"]
                cv2.fillPoly(img, [(pts * SH).astype(np.int32)], fg["col"], cv2.LINE_AA, 4)
        # borda
        cur = pal[int(t / 2) % len(pal)]
        wob = 0.0
        if fr["t_clear"] is not None and not ex:
            lv = fr["lvl"]
            kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.6) if fr["t_lastpop"] is not None else 0.0
            wob = [2, 6, 10, 15][lv] + 10 * kick
            if lv >= 3:
                wob += 10 * min(1.0, (t - fr["t_lastpop"]) / 1.2) ** 2
        if not ex:
            pts = (ar.dense(fr["s"], wob, t) * SH).astype(np.int32)
            gl = np.zeros_like(img)
            cv2.polylines(gl, [pts], True, cur, 14, cv2.LINE_AA, 4)
            img = cv2.add(img, (cv2.GaussianBlur(gl, (0, 0), 9) * 0.8).astype(np.uint8))
            cv2.polylines(img, [pts], True, (245, 240, 250), 4, cv2.LINE_AA, 4)
        for fg in fr["frags"]:
            if fg["kind"] == "line":
                cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"]); Rm = np.array([[cr, -sr], [sr, cr]])
                cv2.line(img, P(fg["c"] + Rm @ fg["a"]), P(fg["c"] + Rm @ fg["b"]), (245, 240, 250), 4, cv2.LINE_AA, 4)
        f = img.astype(np.float32)
        nleft = int(fr["alive"].sum())
        if not ex:
            put_text(f, f"1 BALL vs {info['nb']} BLOCKS", 40, W / 2, 92, (255, 255, 255), stroke=4, stroke_col=(30, 20, 40))
            put_text(f, f"BLOCKS: {nleft}", 30, W / 2 - 160, 175, (120, 200, 255), stroke=4, stroke_col=(20, 15, 20))
            put_text(f, f"BALLS: {len(fr['pos'])}", 30, W / 2 + 160, 175, cur, stroke=4, stroke_col=(20, 15, 20))
            if fr["t_clear"] is not None:
                put_text(f, "ABOUT TO EXPLODE!", 50, W / 2, ar.bottom + 70, (60, 60, 255), stroke=5, stroke_col=(255, 255, 255),
                         scale=1 + 0.06 * math.sin(t * 14))
            else:
                put_text(f, "Follow for more!", 30, W / 2, ar.bottom + 62, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40))
        img = np.clip(f, 0, 255).astype(np.uint8)
        if ex:
            te = t - fr["t_expl"]
            fl = max(0.0, 0.85 * (1 - te / 0.35))
            if fl > 0:
                img = cv2.addWeighted(img, 1 - fl, np.full_like(img, 255), fl, 0)
            if te > 0.6:
                k = min(1.0, (te - 0.6) / 0.4)
                img = (img.astype(np.float32) * (1 - 0.55 * k)).astype(np.uint8)
                f = img.astype(np.float32)
                sc = ease_out_back(min(1.0, (te - 0.7) / 0.4)) if te > 0.7 else 0.05
                blit(f, *emoji("\U0001F4A5", int(150 * max(0.1, sc))), int(W / 2 - 75 * max(0.1, sc)), int(520 - 75 * max(0.1, sc)))
                put_text(f, "ALL BROKEN!", 66, W / 2, 680, cur, stroke=6, stroke_col=(255, 255, 255), scale=max(0.05, sc))
                if te > 1.3:
                    a_ = min(1, (te - 1.3) / 0.3)
                    put_text(f, "Liked it? FOLLOW US!", 44, W / 2, 790, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40), alpha=a_)
                    put_text(f, "Share it with a friend", 32, W / 2, 852, (200, 230, 255), stroke=3, stroke_col=(30, 20, 40), alpha=a_)
                    put_text(f, "New satisfying videos every day", 28, W / 2, 900, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40), alpha=a_)
                img = np.clip(f, 0, 255).astype(np.uint8)
            shake = max(0.0, 18 * (1 - te / 0.7))
        elif fr["t_clear"] is not None:
            lv = fr["lvl"]
            kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.5) if fr["t_lastpop"] is not None else 0.0
            shake = [1.0, 2.5, 4.0, 6.0][lv] + 7 * kick
            if lv >= 3:
                shake += 6 * min(1.0, (t - fr["t_lastpop"]) / 1.2) ** 2
        else:
            shake = 0
        if shake > 0.5:
            img = cv2.warpAffine(img, np.float32([[1, 0, srng.normal() * shake], [0, 1, srng.normal() * shake]]), (W, H))
        return img


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    cfg = CONFIGS[name]
    od = os.path.join(HERE, f"out_{name}"); os.makedirs(od, exist_ok=True)
    if mode == "tune":
        lo, hi = 0.2, 3.0
        for it in range(7):
            k = (lo + hi) / 2
            _, _, st, info = simulate(cfg, k, keep=False)
            tc = st["t_clear"]
            print(name, "k", round(k, 3), "quadrados", info["nb"], "limpou", tc and round(tc, 1), flush=True)
            if tc is None or tc > 33:
                hi = k
            else:
                lo = k
            if tc is not None and 30.5 <= tc <= 33:
                break
        sys.exit()
    k = float(sys.argv[3])
    frames, audio, st, info = simulate(cfg, k)
    print(name, "duracao", len(frames) / FPS, "limpou", st["t_clear"], "explode", st["t_expl"], "final", st["final"], flush=True)
    rd = Rend(cfg, info); srng = np.random.default_rng(1)
    if mode == "preview":
        want = set(int(x) for x in sys.argv[4:])
        for i, fr in enumerate(frames):
            if i in want:
                cv2.imwrite(f"{od}/p_{i:04d}.png", rd.render(fr, st, srng))
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", f"{od}/v_tmp.mp4"], stdin=subprocess.PIPE)
        for fr in frames:
            enc.stdin.write(rd.render(fr, st, srng).tobytes())
        enc.stdin.close(); enc.wait()
        dur = len(frames) / FPS
        make_audio(sorted(audio, key=lambda e: e[0]), dur, f"{od}/audio.wav", cfg.get("scale", "maior"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{od}/v_tmp.mp4", "-i", f"{od}/audio.wav", "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(dur), f"{od}/tmp.mp4"], check=True)
        os.replace(f"{od}/tmp.mp4", f"{od}/{name}.mp4")
        print("ok", f"{od}/{name}.mp4", flush=True)
