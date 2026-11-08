import sys, math, json, subprocess, os
import numpy as np, cv2
os.environ.setdefault("MULT", "3")
import sim3 as sim2
from sim3 import W, H, FPS, NF, C, RB, RA, TH, SPAN, GEAR_R, COLORS, arcs_at
R0 = sim2.R0
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



def render(fr, glows, shake_rng):
    t = fr["t"]
    img = np.zeros((H, W, 3), np.uint8)
    m, delta, arcs = arcs_at(t)
    exploded = fr["exploded"]
    # brilho do estouro
    gl_active = []
    for (t0, gc, ci) in glows:
        if t0 - 0.02 <= t < t0 + 1.0:
            k = 1.0 if t < t0 + 0.15 else max(0.0, 1 - (t - t0 - 0.15) / 0.85)
            gl_active.append((gc, ci, k, t - t0))
            add_glow(img, gc, COLORS[ci], k * 0.9)
    pos, col = fr["pos"], fr["col"]
    f = img.astype(np.float32)
    ip = np.round(pos).astype(np.int32)
    for k in range(len(pos)):
        x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
        pre, inv = SPR[col[k]]
        if x0 < 0 or y0 < 0 or x0 + SPR_S > W or y0 + SPR_S > H:
            if x0 + SPR_S <= 0 or y0 + SPR_S <= 0 or x0 >= W or y0 >= H:
                continue
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + SPR_S), min(H, y0 + SPR_S)
            sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
            f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * inv[sl] + pre[sl]
            continue
        reg = f[y0:y0 + SPR_S, x0:x0 + SPR_S]
        reg *= inv
        reg += pre
    img = np.clip(f, 0, 255).astype(np.uint8)

    # recipiente
    cont_alpha = 1.0
    if fr["t_restart"] and not exploded and fr["t_expl"] is None:
        pass
    restart_t = fr.get("restart_t0")
    if restart_t is not None and t - restart_t < 0.5:
        cont_alpha = max(0.0, (t - restart_t) / 0.5)
    wob = 0.0
    if fr["t_over"] is not None and not exploded:
        lv = fr["lvl"]
        kick = 0.0
        if fr["t_lastpop"] is not None:
            kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.6)
        kick0 = max(0.0, 1 - (t - fr["t_over"]) / 0.8)   # balançada quando enche
        wob = [3, 6, 10, 15][lv] + 10 * kick + 8 * kick0
        if lv >= 3:
            wob += 10 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
    R = fr["R"]
    if not exploded and cont_alpha > 0:
        ov = img.copy()
        if wob > 0:
            angs = np.linspace(0, 2 * math.pi, 240, endpoint=False)
            rr = R + wob * (np.sin(3 * angs + t * 23) * 0.6 + np.sin(5 * angs - t * 31) * 0.4)
            pts = np.stack([C[0] + rr * np.cos(angs), C[1] + rr * np.sin(angs)], 1)
            cv2.polylines(ov, [(pts * SH).astype(np.int32)], True, (255, 255, 255), 3, cv2.LINE_AA, 4)
        else:
            cv2.circle(ov, P(C), int(R * SH), (255, 255, 255), 3, cv2.LINE_AA, 4)
        img = cv2.addWeighted(ov, cont_alpha, img, 1 - cont_alpha, 0) if cont_alpha < 1 else ov
    for fg in fr["frags"]:
        pts = np.array([[fg["c"][0] + fg["r"] * math.cos(a), fg["c"][1] + fg["r"] * math.sin(a)]
                        for a in np.linspace(fg["a0"], fg["a1"], 8)])
        cen = pts.mean(0); cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"])
        pts = (pts - cen) @ np.array([[cr, sr], [-sr, cr]]) + cen
        cv2.polylines(img, [(pts * SH).astype(np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA, 4)

    # pinça (arcos + engrenagem)
    tong_alpha = 1.0
    if exploded:
        tong_alpha = max(0.0, 1 - (t - fr["t_expl"]) / 0.25)
    elif restart_t is not None and t - restart_t < 0.5:
        tong_alpha = cont_alpha
    if tong_alpha > 0:
        ov = img.copy()
        for idx, (c, a0, a1) in enumerate(arcs):
            ax = (int(RA * SH), int(RA * SH))
            d0, d1 = math.degrees(a0), math.degrees(a1)
            cv2.ellipse(ov, P(c), ax, 0, d0, d1, (90, 90, 90), int(TH + 3), cv2.LINE_AA, 4)
            cv2.ellipse(ov, P(c), ax, 0, d0, d1, (178, 178, 178), int(TH), cv2.LINE_AA, 4)
            cv2.ellipse(ov, P(c), (int((RA - 4) * SH),) * 2, 0, d0, d1, (205, 205, 205), 2, cv2.LINE_AA, 4)
            for q in np.linspace(0.12, 0.95, 6):
                a = a0 + SPAN * q
                rp = c + (RA + 4) * np.array([math.cos(a), math.sin(a)])
                cv2.circle(ov, P(rp), int(1.6 * SH), (120, 120, 120), -1, cv2.LINE_AA, 4)
            if idx == 0:
                draw_text_on_arc(ov, c, a0 + SPAN * 0.5, 0.11)
        gp = gear_poly(C, fr["gear_ang"])
        cv2.fillPoly(ov, [gp], (215, 215, 215), cv2.LINE_AA, 4)
        cv2.polylines(ov, [gp], True, (90, 90, 90), 1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(C), int(10 * SH), (150, 150, 150), -1, cv2.LINE_AA, 4)
        cv2.circle(ov, P(C), int(5 * SH), (70, 70, 70), -1, cv2.LINE_AA, 4)
        img = cv2.addWeighted(ov, tong_alpha, img, 1 - tong_alpha, 0) if tong_alpha < 1 else ov
    # onda de choque do estouro
    for (gc, ci, k, dtt) in gl_active:
        if k > 0.05:
            ov = img.copy()
            cv2.circle(ov, P(gc), int((14 + dtt * 160) * SH), tuple(int(min(255, x * 0.6 + 110)) for x in COLORS[ci]), 3, cv2.LINE_AA, 4)
            img = cv2.addWeighted(ov, k * 0.8, img, 1 - k * 0.8, 0)
    if exploded:
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
        M = np.float32([[1, 0, shake_rng.normal() * shake], [0, 1, shake_rng.normal() * shake]])
        img = cv2.warpAffine(img, M, (W, H))
    return img


if __name__ == "__main__":
    seed = int(sys.argv[1])
    preview = [int(x) for x in sys.argv[2:]]
    frames, audio, glows, st = sim2.simulate(seed)
    # marca o instante do recomeço (para o fade-in)
    r0 = None
    for i, fr in enumerate(frames):
        if i and frames[i - 1]["exploded"] and not fr["exploded"]:
            r0 = fr["t"]
        fr["restart_t0"] = r0
    srng = np.random.default_rng(1)
    if r0 is not None:
        later = [g[0] for g in glows if g[0] > r0]
        t_end = (later[0] + 1.4) if later else r0 + 4.0
        frames = [f for f in frames if f["t"] < t_end]
        audio = [a for a in audio if a[0] < t_end]
    print("duracao", len(frames) / FPS, "explosao", st.get("t_expl"), flush=True)
    if preview:
        for fi in preview:
            cv2.imwrite(f"out/v2_{fi:04d}.png", render(frames[fi], glows, srng))
        print("preview ok")
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                "-pix_fmt", "yuv420p", "out/v2_video_tmp.mp4"], stdin=subprocess.PIPE)
        for i, fr in enumerate(frames):
            enc.stdin.write(render(fr, glows, srng).tobytes())
        enc.stdin.close(); enc.wait()
        json.dump(sorted(audio), open("out/audio_events.json", "w"))
        print("ok", enc.returncode)
