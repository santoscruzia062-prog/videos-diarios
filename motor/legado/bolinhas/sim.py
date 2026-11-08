"""Bolinhas que se multiplicam: quando a pinça fecha o círculo, nasce um lote maior de
bolinhas de uma nova cor (com tique-taque); quando lota, o recipiente explode."""
import sys, math, subprocess, json
import numpy as np
import cv2
from scipy.spatial import cKDTree

W, H, FPS = 720, 1280, 30
DUR = 45.0
NF = int(DUR * FPS)
C = np.array([360.0, 640.0])
R0 = 300.0          # raio do recipiente
RB = 9.0            # raio da bolinha
RA = 95.0           # raio dos arcos da pinça
TH = 15.0           # espessura dos arcos
SPAN = math.radians(200)
GEAR_R = 24.0
SUB = 3
DT = 1.0 / (FPS * SUB)

PREVIEW = [int(x) for x in sys.argv[1:]]  # frames para pré-visualizar (sem render completo)

# ---------- roteiro ----------
COLORS = [  # BGR
    (230, 140, 30),   # azul
    (40, 230, 235),   # amarelo
    (225, 60, 175),   # roxo
    (40, 90, 245),    # laranja-avermelhado
    (70, 215, 60),    # verde
    (210, 60, 235),   # magenta
    (235, 215, 40),   # ciano
    (40, 170, 245),   # dourado
    (220, 50, 110),   # violeta
    (60, 40, 225),    # vermelho
    (40, 235, 170),   # lima
    (190, 120, 255),  # rosa
    (180, 210, 40),   # turquesa
    (20, 120, 250),   # laranja
    (60, 235, 90),    # verde (recomeço)
]
BATCH = [3, 5, 9, 15, 25, 38, 56, 78, 104, 134, 170, 205]
EVENTS = [(1.5 + 3.0 * k, BATCH[k], COLORS[k + 1]) for k in range(len(BATCH))]   # ultimo em 34.5
OVER_T = 37.5                 # lote final que lota
EVENTS.append((OVER_T, 280, COLORS[13]))
EXPLODE_T = 40.0
RESTART_T = 42.2
EVENTS.append((44.0, 3, COLORS[0]))
PRESS_TIMES = [e[0] for e in EVENTS]

rng = np.random.default_rng(7)

# ---------- utilidades de desenho ----------
def make_ball_sprite(color, r=RB, ss=6):
    s = int(math.ceil(r)) * 2 + 4
    S = s * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / ss - s / 2 + 0.5 / ss
    d = np.sqrt(xx ** 2 + yy ** 2)
    a = np.clip((r - d) * ss / ss + 0.5, 0, 1)  # cobertura
    col = np.array(color, np.float32)
    # sombreamento: luz vindo do alto-esquerda
    lx, ly = -0.45, -0.55
    nz = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    shade = 0.55 + 0.55 * np.clip((xx / r) * lx + (yy / r) * ly + nz * 0.7, 0, 1)
    rgb = col[None, None, :] * shade[..., None]
    rim = np.clip((d / r - 0.82) / 0.18, 0, 1)[..., None]
    rgb = rgb * (1 - 0.45 * rim)
    hd = np.sqrt((xx / r + 0.38) ** 2 + (yy / r + 0.42) ** 2)
    hl = np.clip(1 - hd / 0.32, 0, 1) ** 1.6
    rgb = rgb + (255 - rgb) * hl[..., None] * 0.9
    rgb = np.clip(rgb, 0, 255)
    rgba = np.dstack([rgb, a * 255]).astype(np.float32)
    small = cv2.resize(rgba, (s, s), interpolation=cv2.INTER_AREA)
    al = small[..., 3:4] / 255.0
    return (small[..., :3] * al).astype(np.float32), (1 - al).astype(np.float32)


SPR = [make_ball_sprite(c) for c in COLORS]
SPR_S = SPR[0][0].shape[0]
HALF = SPR_S // 2


def make_glow(r):
    s = int(r * 2)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
    d = np.sqrt(xx ** 2 + yy ** 2) / r
    return np.clip(1 - d, 0, 1) ** 2


GLOW = make_glow(170)


def add_glow(img, center, color, k):
    if k <= 0.01:
        return
    g = GLOW
    s = g.shape[0]
    x0, y0 = int(center[0] - s / 2), int(center[1] - s / 2)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + s), min(H, y0 + s)
    if xa >= xb or ya >= yb:
        return
    gg = g[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * k
    reg = img[ya:yb, xa:xb].astype(np.float32)
    col = np.array(color, np.float32)
    reg = reg + gg * (col * 0.8 + 60)
    img[ya:yb, xa:xb] = np.clip(reg, 0, 255).astype(np.uint8)


SH = 16  # shift p/ subpixel
def P(p):
    return (int(round(p[0] * SH)), int(round(p[1] * SH)))


def gear_poly(center, ang, r=GEAR_R, teeth=12):
    pts = []
    for i in range(teeth * 4):
        a = ang + i * (2 * math.pi / (teeth * 4))
        rr = r if (i % 4) in (0, 1) else r * 0.78
        pts.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a)))
    return np.array([[int(x * SH), int(y * SH)] for x, y in pts], np.int32)


# letras do logo sobre o arco
LOGO = "NewsLab"
CHAR = {}
for ch in set(LOGO):
    m = np.zeros((26, 26), np.uint8)
    (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
    cv2.putText(m, ch, (13 - tw // 2, 13 + th // 2), cv2.FONT_HERSHEY_DUPLEX, 0.42, 255, 1, cv2.LINE_AA)
    CHAR[ch] = m


def draw_text_on_arc(img, c, a_mid, step):
    n = len(LOGO)
    for i, ch in enumerate(LOGO):
        a = a_mid + (i - (n - 1) / 2) * step
        px, py = c[0] + RA * math.cos(a), c[1] + RA * math.sin(a)
        rot = math.degrees(a) + 90 if step > 0 else math.degrees(a) - 90
        M = cv2.getRotationMatrix2D((13, 13), -rot, 1.0)
        m = cv2.warpAffine(CHAR[ch], M, (26, 26), flags=cv2.INTER_LINEAR).astype(np.float32) / 255 * 0.85
        x0, y0 = int(px) - 13, int(py) - 13
        if x0 < 0 or y0 < 0 or x0 + 26 > W or y0 + 26 > H:
            continue
        reg = img[y0:y0 + 26, x0:x0 + 26].astype(np.float32)
        reg = reg * (1 - m[..., None]) + np.array([70, 70, 70], np.float32) * m[..., None]
        img[y0:y0 + 26, x0:x0 + 26] = reg.astype(np.uint8)


# ---------- mecanismo ----------
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


ALPHA_OPEN = 1.65


def alpha_at(t):
    """abertura da pinça: 0 = círculo fechado"""
    best = ALPHA_OPEN
    for te in PRESS_TIMES:
        if te - 1.3 <= t < te:            # fechando (acelera no fim -> "estalo")
            x = (t - (te - 1.3)) / 1.3
            best = min(best, ALPHA_OPEN * (1 - x ** 2.2))
        elif te <= t < te + 0.45:         # segura fechado
            best = 0.0
        elif te + 0.45 <= t < te + 1.5:   # abre
            best = min(best, ALPHA_OPEN * ease((t - te - 0.45) / 1.05))
    return best


def theta_at(t):
    return -1.2 + t * (2 * math.pi / 7.5)


def gear_at(t):
    return C + np.array([28 * math.cos(0.35 * t + 1), 34 * math.sin(0.27 * t)])


def arcs_at(t):
    g = gear_at(t)
    th = theta_at(t)
    al = alpha_at(t)
    out = []
    for sgn in (+1, -1):
        phi = th + sgn * al
        c = g + RA * np.array([math.cos(phi), math.sin(phi)])
        a_g = phi + math.pi  # ângulo do ponto da engrenagem visto do centro do arco
        if sgn > 0:
            a0 = a_g
        else:
            a0 = a_g - SPAN
        out.append((c, a0))
    return g, out


# ---------- estado ----------
pos = np.zeros((0, 2))
prev = np.zeros((0, 2))
col = np.zeros(0, np.int32)

def spawn(n, center, ci, vmin=60, vmax=260, spread=18):
    global pos, prev, col
    ang = rng.uniform(0, 2 * math.pi, n)
    rad = np.sqrt(rng.uniform(0, 1, n)) * spread
    p = center + np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
    sp = rng.uniform(vmin, vmax, n)
    v = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
    pos = np.vstack([pos, p])
    prev = np.vstack([prev, p - v * DT])
    col = np.concatenate([col, np.full(n, ci, np.int32)])

# bolinha inicial
_g, _arcs = arcs_at(1.5)
pos = np.array([_arcs[0][0] + np.array([-6.0, 4.0])])
prev = pos.copy()
col = np.array([0], np.int32)

audio_ev = []   # (tempo, tipo, volume)

# agenda de surgimentos (pulsos de tique-taque)
spawn_q = []    # (t, n, center_func_key, color_idx)
for k, (te, n, c) in enumerate(EVENTS):
    ci = COLORS.index(c)
    dur = min(1.2, 0.25 + n * 0.004)
    pulses = max(1, min(int(round(dur / 0.1)), n))
    base = n // pulses
    rest = n - base * pulses
    for p in range(pulses):
        cnt = base + (1 if p < rest else 0)
        spawn_q.append([te + 0.06 + p * 0.1, cnt, k, ci])
spawn_q.sort()

# fundo: tique-taque do relógio a cada 0.5s (acelera quando está lotando)
tt = 0.5
tick_i = 0
while tt < DUR:
    if EXPLODE_T <= tt < RESTART_T:
        tt += 0.5
        continue
    audio_ev.append((tt, "tick" if tick_i % 2 == 0 else "tock", 0.22))
    tick_i += 1
    if OVER_T + 0.6 < tt < EXPLODE_T:
        frac = (tt - OVER_T) / (EXPLODE_T - OVER_T)
        tt += max(0.07, 0.5 * (1 - frac) ** 1.5)
    else:
        tt += 0.5
for te in PRESS_TIMES:
    audio_ev.append((te, "press", 0.7))
audio_ev.append((EXPLODE_T, "boom", 1.0))
audio_ev.append((OVER_T + 0.5, "rumble", 0.5))
audio_ev.append((RESTART_T + 0.3, "pop", 0.6))

exploded = False
frags = []
R_eff = R0
shake = 0.0
pulse_i = 0
gear_ang = 0.0
spawn_tick_parity = 0

# ---------- física ----------
def step(t, Rc, has_wall, gravity):
    global pos, prev
    n = len(pos)
    if n == 0:
        return
    v = (pos - prev) / DT
    if not exploded and t > 1.55:
        rel = pos - C
        vt = np.stack([-rel[:, 1], rel[:, 0]], 1) * 0.55       # todas giram no mesmo sentido
        v += (vt - v) * min(1.0, 1.6 * DT)
    v *= 0.9995
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    cap = 2500 if exploded else 900
    v = np.where(sp > cap, v / np.maximum(sp, 1e-6) * cap, v)
    newp = pos + v * DT + gravity * DT * DT
    prev = pos
    pos = newp
    for it in range(2):
        # bolinha x bolinha
        tree = cKDTree(pos)
        pr = tree.query_pairs(2 * RB, output_type="ndarray")
        if len(pr):
            i, j = pr[:, 0], pr[:, 1]
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d, axis=1)
            dist = np.maximum(dist, 1e-4)
            ov = (2 * RB - dist) / dist * 0.5 * 0.9
            corr = d * ov[:, None]
            dp = np.zeros_like(pos)
            np.add.at(dp, i, -corr)
            np.add.at(dp, j, corr)
            pos += dp
        if not exploded and (t >= 1.5 or len(pos) > 1):
            # pinça
            g, arcs = arcs_at(t)
            for c, a0 in arcs:
                d = pos - c
                dist = np.linalg.norm(d, axis=1)
                ang = np.arctan2(d[:, 1], d[:, 0])
                relA = (ang - a0) % (2 * math.pi)
                inside = relA <= SPAN
                gap = np.abs(dist - RA)
                hit = inside & (gap < RB + TH / 2)
                if hit.any():
                    sgn = np.sign(dist[hit] - RA)
                    sgn[sgn == 0] = 1
                    target = RA + sgn * (RB + TH / 2)
                    pos[hit] = c + d[hit] / np.maximum(dist[hit], 1e-4)[:, None] * target[:, None]
                # ponta livre
                ae = a0 + SPAN if a0 == arcs[0][1] else a0
                e = c + RA * np.array([math.cos(ae), math.sin(ae)])
                de = pos - e
                dd = np.linalg.norm(de, axis=1)
                h2 = dd < RB + TH / 2
                if h2.any():
                    pos[h2] = e + de[h2] / np.maximum(dd[h2], 1e-4)[:, None] * (RB + TH / 2)
            dg = pos - g
            dd = np.linalg.norm(dg, axis=1)
            h3 = dd < RB + GEAR_R
            if h3.any():
                pos[h3] = g + dg[h3] / np.maximum(dd[h3], 1e-4)[:, None] * (RB + GEAR_R)
        if has_wall:
            d = pos - C
            dist = np.linalg.norm(d, axis=1)
            out = dist > Rc - RB
            if out.any():
                pos[out] = C + d[out] / dist[out][:, None] * (Rc - RB)


# ---------- render ----------
def render(t, glow_list, flash, cont_alpha, wob, tong_alpha):
    img = np.zeros((H, W, 3), np.uint8)
    g, arcs = arcs_at(t)
    # glow por baixo
    for (gc, gcol, k) in glow_list:
        add_glow(img, gc, gcol, k)
    # bolinhas
    f = img.astype(np.float32)
    ip = np.round(pos).astype(np.int32)
    for k in range(len(pos)):
        x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
        if x0 < 0 or y0 < 0 or x0 + SPR_S > W or y0 + SPR_S > H:
            if x0 + SPR_S <= 0 or y0 + SPR_S <= 0 or x0 >= W or y0 >= H:
                continue
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + SPR_S), min(H, y0 + SPR_S)
            pre, inv = SPR[col[k]]
            sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * inv[sl] + pre[sl]
            continue
        pre, inv = SPR[col[k]]
        reg = f[y0:y0 + SPR_S, x0:x0 + SPR_S]
        reg *= inv
        reg += pre
    img = np.clip(f, 0, 255).astype(np.uint8)
    # recipiente
    if cont_alpha > 0 and not exploded:
        ov = img.copy()
        if wob > 0:
            angs = np.linspace(0, 2 * math.pi, 240, endpoint=False)
            rr = R_eff + wob * (np.sin(3 * angs + t * 23) * 0.6 + np.sin(5 * angs - t * 31) * 0.4)
            pts = np.stack([C[0] + rr * np.cos(angs), C[1] + rr * np.sin(angs)], 1)
            cv2.polylines(ov, [(pts * SH).astype(np.int32)], True, (255, 255, 255), 3, cv2.LINE_AA, 4)
        else:
            cv2.circle(ov, P(C), int(R_eff * SH), (255, 255, 255), 3, cv2.LINE_AA, 4)
        img = cv2.addWeighted(ov, cont_alpha, img, 1 - cont_alpha, 0) if cont_alpha < 1 else ov
    for fr in frags:
        pts = np.array([[fr["c"][0] + fr["r"] * math.cos(a), fr["c"][1] + fr["r"] * math.sin(a)]
                        for a in np.linspace(fr["a0"], fr["a1"], 8)])
        rot = fr["rot"]
        cen = pts.mean(0)
        cr, sr = math.cos(rot), math.sin(rot)
        pts = (pts - cen) @ np.array([[cr, sr], [-sr, cr]]) + cen
        cv2.polylines(img, [(pts * SH).astype(np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA, 4)
    # pinça
    if tong_alpha > 0:
        ov = img.copy()
        for idx, (c, a0) in enumerate(arcs):
            ax = (int(RA * SH), int(RA * SH))
            cv2.ellipse(ov, P(c), ax, 0, math.degrees(a0), math.degrees(a0 + SPAN), (95, 95, 95), int(TH + 3), cv2.LINE_AA, 4)
            cv2.ellipse(ov, P(c), ax, 0, math.degrees(a0), math.degrees(a0 + SPAN), (178, 178, 178), int(TH), cv2.LINE_AA, 4)
            cv2.ellipse(ov, P(c), (int((RA - 4) * SH),) * 2, 0, math.degrees(a0), math.degrees(a0 + SPAN), (205, 205, 205), 2, cv2.LINE_AA, 4)
            # rebites
            for q in np.linspace(0.12, 0.95, 6):
                a = a0 + SPAN * q
                rp = c + (RA + 4) * np.array([math.cos(a), math.sin(a)])
                cv2.circle(ov, P(rp), int(1.6 * SH), (120, 120, 120), -1, cv2.LINE_AA, 4)
            if idx == 0:
                draw_text_on_arc(ov, c, a0 + SPAN * 0.5, 0.115)
        gp = gear_poly(g, gear_ang)
        cv2.fillPoly(ov, [gp], (215, 215, 215), cv2.LINE_AA, 4)
        cv2.polylines(ov, [gp], True, (90, 90, 90), 1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(g), int(10 * SH), (150, 150, 150), -1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(g), int(5 * SH), (70, 70, 70), -1, cv2.LINE_AA, 4)
        img = cv2.addWeighted(ov, tong_alpha, img, 1 - tong_alpha, 0) if tong_alpha < 1 else ov
    # anel brilhante quando "pressiona"
    for (gc, gcol, k) in glow_list:
        if k > 0.05:
            ov = img.copy()
            cv2.circle(ov, P(gc), int((RA + (1 - k) * 40) * SH), tuple(int(min(255, x * 0.6 + 110)) for x in gcol), 4, cv2.LINE_AA, 4)
            img = cv2.addWeighted(ov, min(1, k), img, 1 - min(1, k), 0)
    if flash > 0:
        img = cv2.addWeighted(img, 1 - flash, np.full_like(img, 255), flash, 0)
    if shake > 0.5:
        M = np.float32([[1, 0, rng.normal() * shake], [0, 1, rng.normal() * shake]])
        img = cv2.warpAffine(img, M, (W, H))
    return img


# ---------- laço principal ----------
enc = None
if not PREVIEW:
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-pix_fmt", "yuv420p", "out/video_tmp.mp4"], stdin=subprocess.PIPE)

ring_center_at_press = {}
last_gear_step = -1
maxf = max(PREVIEW) + 1 if PREVIEW else NF
for fi in range(maxf):
    t = fi / FPS
    # engrenagem: dá um "passo" a cada tique
    nstep = int(t / 0.5)
    frac = (t % 0.5) / 0.5
    gear_ang = (nstep + ease(min(1, frac * 5))) * math.radians(30)

    # lotação -> recipiente estufa
    if not exploded:
        need = RB * math.sqrt(len(pos) / 0.86)
        target = max(R0, need) if t >= OVER_T else R0
        if t >= RESTART_T:
            target = R0
        R_eff += (target - R_eff) * 0.08
    wob = 0.0
    if OVER_T + 0.5 <= t < EXPLODE_T:
        wob = 3 + 9 * ((t - OVER_T - 0.5) / (EXPLODE_T - OVER_T - 0.5)) ** 2

    # explosão
    if not exploded and t >= EXPLODE_T and t < RESTART_T:
        exploded = True
        d = pos - C
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
        sp = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / R_eff)
        v = d / dist * sp + rng.normal(0, 120, pos.shape)
        prev = pos - v * DT
        nfr = 22
        for k in range(nfr):
            a0 = k * 2 * math.pi / nfr
            a1 = a0 + 2 * math.pi / nfr * 0.85
            am = (a0 + a1) / 2
            frags.append(dict(c=C.copy(), r=R_eff, a0=a0, a1=a1, rot=0.0,
                              v=np.array([math.cos(am), math.sin(am)]) * rng.uniform(500, 1100),
                              w=rng.uniform(-6, 6)))
    if exploded and t >= RESTART_T:
        exploded = False
        frags = []
        pos = np.zeros((0, 2)); prev = np.zeros((0, 2)); col = np.zeros(0, np.int32)
        R_eff = R0
    if exploded:
        shake = max(0.0, 18 * (1 - (t - EXPLODE_T) / 0.7))
    elif OVER_T + 0.5 <= t < EXPLODE_T:
        shake = 3 * ((t - OVER_T - 0.5) / (EXPLODE_T - OVER_T - 0.5)) ** 2
    else:
        shake = 0.0

    # recomeço: 1 bolinha verde
    if abs(t - (RESTART_T + 0.3)) < 0.5 / FPS and not exploded:
        spawn(1, C + np.array([-200.0, -140.0]), 14, 80, 120, 1)

    # surgimentos agendados
    g, arcs = arcs_at(t)
    while pulse_i < len(spawn_q) and spawn_q[pulse_i][0] <= t:
        ts, n, k, ci = spawn_q[pulse_i]
        te = EVENTS[k][0]
        if k not in ring_center_at_press:
            ring_center_at_press[k] = arcs[0][0].copy()
        if n > 0:
            spawn(n, arcs[0][0], ci, 80, 240 + n * 0.6, spread=min(60, 10 + n * 0.25))
        audio_ev.append((ts, "stick" if spawn_tick_parity % 2 == 0 else "stock", 0.6))
        spawn_tick_parity += 1
        pulse_i += 1

    # brilho do "pressionar"
    glow_list = []
    for k, (te, n, c) in enumerate(EVENTS):
        if te - 0.05 <= t < te + 1.4:
            kk = 1.0 if t < te + 0.35 else max(0, 1 - (t - te - 0.35) / 1.05)
            if t < te:
                kk = 0.6
            glow_list.append((arcs[0][0] if t < te + 0.45 else ring_center_at_press.get(k, arcs[0][0]), c, kk))

    flash = 0.0
    if exploded:
        flash = max(0.0, 0.85 * (1 - (t - EXPLODE_T) / 0.35))
    cont_alpha = 1.0
    if t >= RESTART_T:
        cont_alpha = min(1.0, (t - RESTART_T) / 0.5)
    tong_alpha = 1.0
    if t >= EXPLODE_T:
        tong_alpha = max(0.0, 1 - (t - EXPLODE_T) / 0.25) if t < RESTART_T else min(1.0, (t - RESTART_T) / 0.5)

    grav = np.array([0.0, 900.0]) if exploded else np.array([0.0, 0.0])
    for s in range(SUB):
        step(t + s * DT, R_eff, not exploded, grav)
    if exploded:
        keep = (pos[:, 0] > -50) & (pos[:, 0] < W + 50) & (pos[:, 1] < H + 50) & (pos[:, 1] > -400)
        pos, prev, col = pos[keep], prev[keep], col[keep]
        for fr in frags:
            fr["c"] = fr["c"] + fr["v"] / FPS
            fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS
            fr["rot"] += fr["w"] / FPS

    if PREVIEW and fi in PREVIEW:
        img = render(t, glow_list, flash, cont_alpha, wob, tong_alpha)
        cv2.imwrite(f"out/prev_{fi:04d}.png", img)
        print("frame", fi, "t=%.2f" % t, "bolinhas", len(pos), flush=True)
    elif enc:
        img = render(t, glow_list, flash, cont_alpha, wob, tong_alpha)
        enc.stdin.write(img.tobytes())
        if fi % 90 == 0:
            print("frame", fi, "bolinhas", len(pos), flush=True)

if enc:
    enc.stdin.close()
    enc.wait()
    json.dump(sorted(audio_ev), open("out/audio_events.json", "w"))
    print("ok", enc.returncode)
