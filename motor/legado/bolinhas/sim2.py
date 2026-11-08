"""v2 — mecanismo igual à referência: dois arcos presos na engrenagem giram sem parar em
sentidos opostos e passam um pelo outro. Quando as pontas se encontram forma-se um anel;
as bolinhas presas nele são espremidas contra a engrenagem e estouram em mais bolinhas
(de uma cor nova), com tique-taque a cada surgimento. Quando lota, o recipiente explode."""
import sys, math, subprocess, json
import numpy as np
import cv2
from scipy.spatial import cKDTree

W, H, FPS = 720, 1280, 30
DUR = 45.0
NF = int(DUR * FPS)
C = np.array([360.0, 640.0])   # centro do recipiente = engrenagem
R0 = 292.0
RB = 9.0
RA = 100.0
TH = 14.0
SPAN = math.radians(205)
GEAR_R = 24.0
SUB = 3
DT = 1.0 / (FPS * SUB)
T_CYC = 2.1                    # um encontro das pontas a cada 2,1 s (como na referência)
W_REL = math.pi / T_CYC
OMEGA = math.radians(45) / T_CYC
T1 = 0.9                       # primeiro encontro das pontas
CAP = 870                      # lotação
MULT = int(__import__("os").environ.get("MULT", "4"))

COLORS = [  # BGR
    (230, 140, 30), (40, 230, 235), (225, 60, 175), (40, 90, 245), (70, 215, 60),
    (210, 60, 235), (235, 215, 40), (40, 170, 245), (220, 50, 110), (60, 40, 225),
    (40, 235, 170), (190, 120, 255), (180, 210, 40), (20, 120, 250), (60, 235, 90),
    (245, 95, 70), (110, 230, 240), (150, 70, 230),
]


def arcs_at(t):
    m = math.pi + OMEGA * (t - T1)
    d = W_REL * (t - T1)
    pa, pb = m + d, m - d
    ca = C + RA * np.array([math.cos(pa), math.sin(pa)])
    cb = C + RA * np.array([math.cos(pb), math.sin(pb)])
    a0a = pa + math.pi                 # arco A: sai da engrenagem e avança (+)
    a0b = pb + math.pi - SPAN          # arco B: chega na engrenagem
    return m, (2 * d) % (2 * math.pi), [(ca, a0a, a0a + SPAN), (cb, a0b, pb + math.pi)]


def align_times():
    out, k = [], 0
    while True:
        t = T1 + k * T_CYC
        if t > DUR:
            return out
        out.append(t)
        k += 1


ALIGN = align_times()


# ======================= SIMULAÇÃO =======================
def simulate(seed, keep=True, verbose=False):
    rng = np.random.default_rng(seed)
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2))
    col = np.zeros(0, np.int32); trapped = np.zeros(0, bool); ghost = np.zeros(0, bool)
    frames, audio = [], []
    state = dict(R=R0, exploded=False, t_over=None, t_expl=None, t_restart=None, ci=0)
    frags = []
    glows = []            # (t0, pos, color_idx)
    spawn_q = []          # [t, n, center, dir, ci]
    cyc_done = set(); pop_done = set()

    def add_balls(n, center, direction, ci, t):
        nonlocal pos, prev, col, trapped, ghost
        base = math.atan2(direction[1], direction[0])
        ang = base + rng.normal(0, 0.45, n)
        rad = rng.uniform(0, 1, n) ** 0.5 * min(40, 6 + n * 0.12)
        p = center + np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1) + rng.normal(0, 2, (n, 2))
        sp = rng.uniform(170, 330, n)
        v = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        col = np.concatenate([col, np.full(n, ci, np.int32)])
        trapped = np.concatenate([trapped, np.zeros(n, bool)])
        ghost = np.concatenate([ghost, np.zeros(n, bool)])

    def place_single(t_align, ci):
        nonlocal pos, prev, col, trapped, ghost
        _, _, arcs = arcs_at(t_align)
        p = arcs[0][0] + np.array([0.0, 0.0])
        pos = p[None].copy(); prev = pos.copy()
        col = np.array([ci], np.int32); trapped = np.zeros(1, bool); ghost = np.ones(1, bool)

    place_single(ALIGN[0], 0)
    t_next_tick = 0.5; tick_i = 0
    gear_ang = 0.0

    for fi in range(NF):
        t = fi / FPS
        m, delta, arcs = arcs_at(t)
        exploded = state["exploded"]

        # --- tique-taque do relógio ---
        while t_next_tick <= t:
            if not (state["t_expl"] and state["t_expl"] <= t_next_tick < (state["t_restart"] or 99)):
                audio.append((t_next_tick, "tick" if tick_i % 2 == 0 else "tock", 0.22))
                tick_i += 1
            if state["t_over"] and not exploded and t_next_tick >= state["t_over"]:
                frac = min(1, (t_next_tick - state["t_over"]) / 1.8)
                t_next_tick += max(0.07, 0.5 * (1 - frac) ** 1.5)
            else:
                t_next_tick += 0.5
        gstep = int(t / 0.5)
        gear_ang = (gstep + min(1, ((t % 0.5) / 0.5) * 5)) * math.radians(30)

        # --- pontas se encontram: anel fechado prende as bolinhas de dentro ---
        if not exploded:
            for k, ta in enumerate(ALIGN):
                if k not in cyc_done and t >= ta:
                    cyc_done.add(k)
                    if len(pos):
                        _, _, ar = arcs_at(ta)
                        dd = np.linalg.norm(pos - ar[0][0], axis=1)
                        inside = dd < RA - TH / 2 - RB * 0.3
                        trapped |= inside
                        ghost[:] = False
                        audio.append((ta, "clink", 0.45))
            # estouro: lente fina demais -> as presas viram MULT bolinhas cada
            for k, ta in enumerate(ALIGN):
                tp = ta + 0.72
                if k not in pop_done and t >= tp and k in cyc_done:
                    pop_done.add(k)
                    nt = int(trapped.sum())
                    if nt:
                        ca, cb = arcs[0][0], arcs[1][0]
                        center = C + 0.5 * ((ca + cb) - 2 * C) * 0.9
                        direction = np.array([math.cos(m), math.sin(m)])
                        keepm = ~trapped
                        pos, prev, col, ghost = pos[keepm], prev[keepm], col[keepm], ghost[keepm]
                        trapped = trapped[keepm]
                        state["ci"] = (state["ci"] + 1) % len(COLORS)
                        ci = state["ci"]
                        n = nt * MULT
                        pulses = max(1, min(14, (n + 3) // 4))
                        base, rest = divmod(n, pulses)
                        for p in range(pulses):
                            spawn_q.append([t + p * 0.075, base + (1 if p < rest else 0), center, direction, ci])
                        glows.append((t, center.copy(), ci))
                        audio.append((t, "press", 0.75))
        # surgimentos (tique-taque a cada um)
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, cen, dr, ci = spawn_q.pop(0)
            if not state["exploded"]:
                add_balls(n, cen, dr, ci, t)
                audio.append((ts, "stick" if len([a for a in audio if a[1] in ("stick", "stock")]) % 2 == 0 else "stock", 0.6))

        # --- lotação ---
        if not exploded and state["t_over"] is None and len(pos) >= CAP:
            state["t_over"] = t
            audio.append((t, "rumble", 0.5))
        if not exploded and state["t_over"] is not None:
            need = RB * math.sqrt(len(pos) / 0.84)
            state["R"] += (max(R0, need) - state["R"]) * 0.08
            if t >= state["t_over"] + 1.8:
                state["exploded"] = exploded = True
                state["t_expl"] = t
                state["t_restart"] = t + 2.3
                audio.append((t, "boom", 1.0))
                d = pos - C
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                sp = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / state["R"])
                v = d / dist * sp + rng.normal(0, 120, pos.shape)
                prev = pos - v * DT
                nfr = 22
                for k in range(nfr):
                    a0 = k * 2 * math.pi / nfr
                    a1 = a0 + 2 * math.pi / nfr * 0.85
                    am = (a0 + a1) / 2
                    frags.append(dict(c=C.copy(), r=state["R"], a0=a0, a1=a1, rot=0.0,
                                      v=np.array([math.cos(am), math.sin(am)]) * rng.uniform(500, 1100),
                                      w=rng.uniform(-6, 6)))
        if exploded and t >= state["t_restart"]:
            state.update(exploded=False, t_over=None, R=R0)
            exploded = False
            frags = []
            nxt = [ta for ta in ALIGN if ta > t + 0.6]
            if nxt:
                place_single(nxt[0], (state["ci"] + 1) % len(COLORS))
                state["ci"] = (state["ci"] + 1) % len(COLORS)
                audio.append((t, "pop", 0.6))
                # ciclos antigos não contam mais
                for k, ta in enumerate(ALIGN):
                    if ta <= t:
                        cyc_done.add(k); pop_done.add(k)

        # --- física ---
        grav = np.array([0.0, 900.0]) if exploded else np.zeros(2)
        for s in range(SUB):
            ts = t + s * DT
            if len(pos) == 0:
                break
            v = (pos - prev) / DT
            v *= 0.9993
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            cap = 2500 if exploded else 700
            v = np.where(spd > cap, v / np.maximum(spd, 1e-6) * cap, v)
            newp = pos + v * DT + grav * DT * DT
            prev = pos; pos = newp
            _, dl, ar = arcs_at(ts)
            for it in range(2):
                tree = cKDTree(pos)
                pr = tree.query_pairs(2 * RB, output_type="ndarray")
                if len(pr):
                    i, j = pr[:, 0], pr[:, 1]
                    d = pos[j] - pos[i]
                    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4)
                    ov = (2 * RB - dist) / dist * 0.5 * 0.85
                    corr = d * ov[:, None]
                    dp = np.zeros_like(pos)
                    np.add.at(dp, i, -corr); np.add.at(dp, j, corr)
                    pos += dp
                if not exploded:
                    solid = ~ghost
                    for idx, (c, a0, a1) in enumerate(ar):
                        d = pos - c
                        dist = np.linalg.norm(d, axis=1)
                        rel = (np.arctan2(d[:, 1], d[:, 0]) - a0) % (2 * math.pi)
                        hit = solid & ~trapped & (rel <= SPAN) & (np.abs(dist - RA) < RB + TH / 2)
                        if hit.any():
                            sg = np.sign(dist[hit] - RA); sg[sg == 0] = 1
                            tgt = RA + sg * (RB + TH / 2)
                            pos[hit] = c + d[hit] / np.maximum(dist[hit], 1e-4)[:, None] * tgt[:, None]
                        # ponta livre
                        ae = a1 if idx == 0 else a0
                        e = c + RA * np.array([math.cos(ae), math.sin(ae)])
                        de = pos - e
                        dd = np.linalg.norm(de, axis=1)
                        h2 = solid & ~trapped & (dd < RB + TH / 2)
                        if h2.any():
                            pos[h2] = e + de[h2] / np.maximum(dd[h2], 1e-4)[:, None] * (RB + TH / 2)
                    # presas: dentro dos dois círculos (lente que vai se fechando)
                    if trapped.any():
                        for (c, a0, a1) in ar:
                            d = pos - c
                            dist = np.linalg.norm(d, axis=1)
                            lim = RA - TH / 2 - RB
                            out = trapped & (dist > lim)
                            if out.any():
                                pos[out] = c + d[out] / dist[out][:, None] * lim
                    dg = pos - C
                    dd = np.linalg.norm(dg, axis=1)
                    h3 = solid & (dd < RB + GEAR_R)
                    if h3.any():
                        pos[h3] = C + dg[h3] / np.maximum(dd[h3], 1e-4)[:, None] * (RB + GEAR_R)
                    # parede (quique)
                    d = pos - C
                    dist = np.linalg.norm(d, axis=1)
                    Rc = state["R"]
                    out = dist > Rc - RB
                    if out.any():
                        nrm = d[out] / dist[out][:, None]
                        pos[out] = C + nrm * (Rc - RB)
                        vv = pos[out] - prev[out]
                        vn = (vv * nrm).sum(1, keepdims=True)
                        vv = vv - np.maximum(vn, 0) * nrm * 1.85
                        prev[out] = pos[out] - vv
        if exploded:
            kp = (pos[:, 0] > -50) & (pos[:, 0] < W + 50) & (pos[:, 1] < H + 50) & (pos[:, 1] > -500)
            pos, prev, col, trapped, ghost = pos[kp], prev[kp], col[kp], trapped[kp], ghost[kp]
            for fr in frags:
                fr["c"] = fr["c"] + fr["v"] / FPS
                fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS
                fr["rot"] += fr["w"] / FPS

        if keep:
            frames.append(dict(t=t, pos=pos.copy(), col=col.copy(), R=state["R"], exploded=exploded,
                               t_over=state["t_over"], t_expl=state["t_expl"], t_restart=state["t_restart"],
                               frags=[dict(f) for f in frags], gear_ang=gear_ang))
        if verbose and fi % 60 == 0:
            print(f"t={t:5.1f} n={len(pos)} trapped={int(trapped.sum())}", flush=True)
    return frames, audio, glows, state


if __name__ == "__main__":
    if sys.argv[1] == "search":
        for seed in range(int(sys.argv[2]), int(sys.argv[3])):
            _, au, gl, st = simulate(seed, keep=False)
            te = [a[0] for a in au if a[1] == "boom"]
            print(seed, "explode", te, "bursts", len(gl), flush=True)
