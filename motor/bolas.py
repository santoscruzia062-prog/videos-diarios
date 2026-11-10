"""BOLINHAS QUE SE MULTIPLICAM — motor com mecanismos e arenas variáveis.
Bolinhas coloridas são espremidas por um mecanismo que gira sem parar (engrenagens ou arcos),
cada esmagada vira mais bolinhas de uma cor nova; enche TUDO, mais 3 rodadas tremendo, explode.
Tique-taque constante + notas musicais (tom, tom, trim, plim). Gancho: "Quantas até explodir?"."""
import sys, os, math, json, subprocess, wave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 720, 1280, 30
SUB = 3
DT = 1.0 / (FPS * SUB)
DUR = 52.0
FONT_B = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
SH = 16

PALETTES = {
    "manchas": [(170, 60, 245), (40, 150, 255), (70, 205, 70), (235, 140, 40), (40, 225, 245), (200, 70, 150)],
    "vivo": [(230, 140, 30), (40, 230, 235), (225, 60, 175), (40, 90, 245), (70, 215, 60), (210, 60, 235),
             (235, 215, 40), (40, 170, 245), (220, 50, 110), (60, 40, 225), (40, 235, 170), (190, 120, 255),
             (180, 210, 40), (20, 120, 250), (245, 95, 70), (110, 230, 240)],
    "neon": [(255, 80, 200), (80, 255, 120), (255, 200, 40), (40, 200, 255), (200, 80, 255), (60, 255, 255),
             (120, 120, 255), (60, 160, 255), (255, 120, 120), (180, 255, 60), (255, 60, 120), (90, 255, 200),
             (255, 160, 60), (150, 90, 255), (60, 255, 160), (255, 230, 120)],
    "doce": [(203, 192, 255), (180, 229, 255), (230, 216, 173), (193, 182, 255), (170, 232, 238), (221, 160, 221),
             (152, 251, 152), (255, 228, 181), (147, 112, 219), (122, 160, 255), (250, 206, 135), (180, 105, 255),
             (238, 238, 175), (130, 220, 255), (200, 160, 255), (160, 240, 200)],
}


# ======================= arena (formato) =======================
def poly_regular(n, apothem, cx, cy, rot):
    R = apothem / math.cos(math.pi / n)
    return [(cx + R * math.cos(rot + 2 * math.pi * k / n), cy + R * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def shape_pts(kind, cy=650):
    if kind == "circle":
        return [(360 + 305 * math.cos(a), cy + 305 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 200, endpoint=False)]
    if kind == "square":
        return [(50, cy - 310), (670, cy - 310), (670, cy + 310), (50, cy + 310)]
    if kind == "hexagon":
        return poly_regular(6, 300, 360, cy, 0)
    if kind == "octagon":
        return poly_regular(8, 305, 360, cy, math.pi / 8)
    if kind == "triangle":
        return poly_regular(3, 178, 360, cy + 60, -math.pi / 2)
    if kind == "heart":
        t = np.linspace(0, 2 * math.pi, 260, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        s = 680 / (x.max() - x.min())
        x = x * s; y = y * s
        x = x - (x.max() + x.min()) / 2 + 360
        y = y - (y.max() + y.min()) / 2 + cy + 10
        return list(zip(x, y))
    if kind == "pentagon":
        return poly_regular(5, 268, 360, cy + 30, -math.pi / 2)
    if kind == "star":
        R1, R2 = 345, 175
        return [(360 + (R1 if k % 2 == 0 else R2) * math.cos(-math.pi / 2 + k * math.pi / 5),
                 cy + 40 + (R1 if k % 2 == 0 else R2) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]
    if kind == "diamond":
        return [(360, cy - 390), (680, cy), (360, cy + 390), (40, cy)]
    if kind == "rounded":
        pts = []
        r = 90; x0, x1, y0, y1 = 50, 670, cy - 330, cy + 330
        for cx_, cy_, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
            for a in np.linspace(a0, a0 + 90, 16):
                pts.append((cx_ + r * math.cos(math.radians(a)), cy_ + r * math.sin(math.radians(a))))
        return pts
    raise ValueError(kind)


class Arena:
    def __init__(self, kind, cy=650):
        self.pts = np.array(shape_pts(kind, cy), np.float64)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [(self.pts * SH).astype(np.int32)], 255, cv2.LINE_8, 4)
        self.mask = mask
        din = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        sd = (din - dout).astype(np.float32)
        gy, gx = np.gradient(cv2.GaussianBlur(sd, (0, 0), 1.5))
        nn = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        self.sd, self.gx, self.gy = sd, (gx / nn).astype(np.float32), (gy / nn).astype(np.float32)
        self.C = np.array([self.pts[:, 0].mean(), self.pts[:, 1].mean()]) if kind != "heart" else \
            np.array([360.0, float(np.unravel_index(np.argmax(din), din.shape)[0])])
        self.area = float((mask > 0).sum())
        self.top = float(self.pts[:, 1].min()); self.bottom = float(self.pts[:, 1].max())

    def inside(self, p):
        x = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        return self.mask[y, x] > 0

    def sample(self, p, s=1.0):
        q = self.C + (p - self.C) / s
        x = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1); y = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        return self.sd[y, x] * s, self.gx[y, x], self.gy[y, x]

    def dense(self, s=1.0, wob=0.0, t=0.0):
        P = self.C + (self.pts - self.C) * s
        out, nr = [], []
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            L = np.linalg.norm(b - a); m = max(1, int(L / 10))
            tv = (b - a) / (L + 1e-9); nv = np.array([tv[1], -tv[0]])
            if self.inside(((a + b) / 2 + nv * 3)[None])[0]:
                nv = -nv
            for j in range(m):
                out.append(a + (b - a) * j / m); nr.append(nv)
        out = np.array(out); nr = np.array(nr)
        if wob > 0:
            k = np.arange(len(out)); nn = len(out)
            out = out + nr * (wob * (0.7 + 0.3 * np.sin(k * 0.35 + t * 25)) * (0.6 + 0.4 * np.sin(k * 2 * math.pi * 6 / nn + t * 9)))[:, None]
        return out


# ======================= mecanismos =======================
class Gears:
    """Duas engrenagens que se encaixam e giram em sentidos opostos; puxam as bolinhas para o
    'aperto' (nip) do lado de entrada, esmagam e cospem do outro lado."""
    kind = "gears"

    def __init__(self, C, Rc=86.0, axis_deg=0.0, omega=1.25, teeth=14, colA=(70, 180, 230), colB=(190, 190, 200)):
        a = math.radians(axis_deg)
        self.ax = np.array([math.cos(a), math.sin(a)])
        self.n_in = np.array([self.ax[1], -self.ax[0]])          # lado de entrada (perpendicular)
        if self.n_in[1] > 0:
            self.n_in = -self.n_in                                # entrada para cima
        self.Rc = Rc; self.tip = Rc + 9; self.root = Rc - 6
        d = Rc * 0.98
        self.C = np.array(C, float)
        self.g = [self.C - self.ax * d, self.C + self.ax * d]
        self.teeth = teeth
        # sinal da rotação: a superfície no aperto anda da entrada para a saída
        self.w = []
        for gc in self.g:
            r = self.C - gc
            v = np.array([-r[1], r[0]])
            s = 1 if v @ (-self.n_in) > 0 else -1
            self.w.append(s * omega)
        self.colA, self.colB = colA, colB
        self.out_pt = self.C - self.n_in * (Rc * 0.42 + 14)

    def angle(self, i, t):
        return self.w[i] * t + (math.pi / self.teeth if i == 1 else 0)

    def collide(self, pos, prev, mask, RB):
        for i, gc in enumerate(self.g):
            d = pos - gc
            dist = np.linalg.norm(d, axis=1)
            lim = self.Rc + RB
            hit = mask & (dist < lim)
            if hit.any():
                nrm = d[hit] / np.maximum(dist[hit], 1e-4)[:, None]
                pos[hit] = gc + nrm * lim
                # arrasto da superfície (as bolinhas são levadas pelos dentes)
                vs = self.w[i] * np.stack([-nrm[:, 1], nrm[:, 0]], 1) * lim
                vv = (pos[hit] - prev[hit]) / DT
                vv += (vs - vv) * 0.35
                prev[hit] = pos[hit] - vv * DT

    def pull(self, pos, v, RB):
        # leve atração para a engrenagem mais próxima (as bolinhas "grudam" e são levadas ao aperto)
        d0 = pos - self.g[0]; d1 = pos - self.g[1]
        n0 = np.linalg.norm(d0, axis=1); n1 = np.linalg.norm(d1, axis=1)
        near0 = n0 < n1
        d = np.where(near0[:, None], d0, d1); n = np.where(near0, n0, n1)
        f = np.clip((n - self.Rc - RB) / 220, 0, 1)[:, None] * 160
        v -= d / np.maximum(n, 1e-4)[:, None] * f * DT

    def pinched(self, pos, RB):
        d0 = np.linalg.norm(pos - self.g[0], axis=1); d1 = np.linalg.norm(pos - self.g[1], axis=1)
        side = (pos - self.C) @ self.n_in
        return (d0 < self.Rc + RB + 7) & (d1 < self.Rc + RB + 7) & (side > -RB)

    def draw(self, img, t, glowk=0.0):
        for i, gc in enumerate(self.g):
            a0 = self.angle(i, t)
            pts = []
            n = self.teeth * 4
            for k in range(n):
                a = a0 + 2 * math.pi * k / n
                rr = self.tip if (k % 4) in (1, 2) else self.root
                pts.append((gc[0] + rr * math.cos(a), gc[1] + rr * math.sin(a)))
            poly = (np.array(pts) * SH).astype(np.int32)
            col = self.colA if i == 0 else self.colB
            cv2.fillPoly(img, [poly], tuple(int(c * 0.55) for c in col), cv2.LINE_AA, 4)
            inner = [(gc[0] + (self.root - 4) * math.cos(a0 + 2 * math.pi * k / 60), gc[1] + (self.root - 4) * math.sin(a0 + 2 * math.pi * k / 60)) for k in range(60)]
            cv2.fillPoly(img, [(np.array(inner) * SH).astype(np.int32)], col, cv2.LINE_AA, 4)
            cv2.polylines(img, [poly], True, (40, 35, 40), 2, cv2.LINE_AA, 4)
            # raios e furos (mostram o giro)
            for k in range(5):
                a = a0 + 2 * math.pi * k / 5
                p = gc + np.array([math.cos(a), math.sin(a)]) * self.Rc * 0.52
                cv2.circle(img, P(p), int(self.Rc * 0.17 * SH), tuple(int(c * 0.45) for c in col), -1, cv2.LINE_AA, 4)
            cv2.circle(img, P(gc), int(self.Rc * 0.30 * SH), tuple(min(255, int(c * 1.2) + 20) for c in col), -1, cv2.LINE_AA, 4)
            cv2.circle(img, P(gc), int(self.Rc * 0.12 * SH), (45, 40, 45), -1, cv2.LINE_AA, 4)
            cv2.ellipse(img, P(gc), (int(self.Rc * 0.86 * SH),) * 2, 0, 200, 250, (255, 255, 255), 2, cv2.LINE_AA, 4)


class Arcs:
    """Dois arcos presos numa engrenagem central giram em sentidos opostos e passam um pelo outro;
    quando as pontas se encontram forma-se um anel, as bolinhas presas são espremidas contra a
    engrenagem e saem multiplicadas (o mecanismo do vídeo favorito do dono)."""
    kind = "arcs"

    def __init__(self, C, RA=100.0, period=2.1, omega_extra=45, colA=(178, 178, 178)):
        self.C = np.array(C, float); self.RA = RA; self.TH = 14.0
        self.SPAN = math.radians(205); self.T = period
        self.W_REL = math.pi / period; self.OMEGA = math.radians(omega_extra) / period
        self.T1 = 0.9; self.GEAR = 24.0; self.col = colA

    def arcs_at(self, t):
        m = math.pi + self.OMEGA * (t - self.T1)
        d = self.W_REL * (t - self.T1)
        pa, pb = m + d, m - d
        ca = self.C + self.RA * np.array([math.cos(pa), math.sin(pa)])
        cb = self.C + self.RA * np.array([math.cos(pb), math.sin(pb)])
        return m, [(ca, pa + math.pi, pa + math.pi + self.SPAN), (cb, pb + math.pi - self.SPAN, pb + math.pi)]

    def draw(self, img, t, glowk=0.0):
        m, arcs = self.arcs_at(t)
        for (c, a0, a1) in arcs:
            ax = (int(self.RA * SH),) * 2
            cv2.ellipse(img, P(c), ax, 0, math.degrees(a0), math.degrees(a1), (90, 90, 90), int(self.TH + 3), cv2.LINE_AA, 4)
            cv2.ellipse(img, P(c), ax, 0, math.degrees(a0), math.degrees(a1), self.col, int(self.TH), cv2.LINE_AA, 4)
            cv2.ellipse(img, P(c), (int((self.RA - 4) * SH),) * 2, 0, math.degrees(a0), math.degrees(a1), (215, 215, 215), 2, cv2.LINE_AA, 4)
        ga = int(t / 0.5) * math.radians(30) + min(1, (t % 0.5) * 10) * math.radians(30)
        pts = []
        for i in range(48):
            a = ga + i * 2 * math.pi / 48
            rr = self.GEAR if (i % 4) in (0, 1) else self.GEAR * 0.78
            pts.append((self.C[0] + rr * math.cos(a), self.C[1] + rr * math.sin(a)))
        gp = (np.array(pts) * SH).astype(np.int32)
        cv2.fillPoly(img, [gp], (215, 215, 215), cv2.LINE_AA, 4)
        cv2.polylines(img, [gp], True, (90, 90, 90), 1, cv2.LINE_AA, 4)
        cv2.circle(img, P(self.C), int(10 * SH), (150, 150, 150), -1, cv2.LINE_AA, 4)
        cv2.circle(img, P(self.C), int(5 * SH), (70, 70, 70), -1, cv2.LINE_AA, 4)


class Rods:
    """Bastões girando (1, 2 ou 4). Cada bolinha que leva uma batida de bastão se divide em mais
    bolinhas da cor da vez — quanto mais bolinhas, mais batidas, mais rápido enche."""
    kind = "rods"

    def __init__(self, C, rods, th=16.0, col=(200, 200, 210), marks=None):
        self.C = np.array(C, float); self.rods = rods; self.th = th; self.col = col; self.marks = marks

    def segs(self, t):
        out = []
        for r in self.rods:
            p = self.C + np.array(r["off"], float)
            a = r.get("phase", 0.0) + r["w"] * t
            u = np.array([math.cos(a), math.sin(a)])
            out.append((p if r.get("half") else p - u * r["L"], p + u * r["L"], p, r["w"]))
        return out

    def collide(self, pos, prev, mask, RB, t):
        hit_all = np.zeros(len(pos), bool)
        for a, b, p, w in self.segs(t):
            ab = b - a; L2 = ab @ ab
            s_ = np.clip(((pos - a) @ ab) / L2, 0, 1)
            cp = a + s_[:, None] * ab
            d = pos - cp; dist = np.linalg.norm(d, axis=1)
            lim = RB + self.th / 2
            hit = mask & (dist < lim)
            if hit.any():
                nrm = d[hit] / np.maximum(dist[hit], 1e-4)[:, None]
                pos[hit] = cp[hit] + nrm * lim
                r = cp[hit] - p
                vs = w * np.stack([-r[:, 1], r[:, 0]], 1)
                vv = (pos[hit] - prev[hit]) / DT
                vn = (vv * nrm).sum(1, keepdims=True)
                vv = vv + nrm * np.maximum(0, (vs * nrm).sum(1, keepdims=True) * 1.2 - vn)
                prev[hit] = pos[hit] - vv * DT
            hit_all |= mask & (dist < lim + 2)
        return hit_all

    def draw(self, img, t, glowk=0.0):
        if self.marks:
            for k in range(12):
                ang = k * math.pi / 6; u = np.array([math.cos(ang), math.sin(ang)])
                r0 = self.marks - (26 if k % 3 == 0 else 16)
                cv2.line(img, P(self.C + u * r0), P(self.C + u * (self.marks - 4)), (240, 235, 245), 6 if k % 3 == 0 else 3, cv2.LINE_AA, 4)
        for a, b, p, w in self.segs(t):
            u = (b - a) / (np.linalg.norm(b - a) + 1e-9); n = np.array([-u[1], u[0]]); h = self.th / 2
            cv2.line(img, P(a), P(b), (55, 50, 60), int(self.th + 4), cv2.LINE_AA, 4)
            cv2.line(img, P(a), P(b), self.col, int(self.th), cv2.LINE_AA, 4)
            cv2.line(img, P(a + n * (h - 3)), P(b + n * (h - 3)), tuple(min(255, c + 45) for c in self.col), 2, cv2.LINE_AA, 4)
            for e in (a, b):
                cv2.circle(img, P(e), int(h * SH), self.col, -1, cv2.LINE_AA, 4)
            ga = w * t
            pts = [(p[0] + (15 if (k % 4) in (0, 1) else 11) * math.cos(ga + k * 2 * math.pi / 40), p[1] + (15 if (k % 4) in (0, 1) else 11) * math.sin(ga + k * 2 * math.pi / 40)) for k in range(40)]
            gp = (np.array(pts) * SH).astype(np.int32)
            cv2.fillPoly(img, [gp], (70, 175, 215), cv2.LINE_AA, 4)
            cv2.circle(img, P(p), int(5 * SH), (50, 45, 50), -1, cv2.LINE_AA, 4)


class Pieces:
    """Peças metálicas genéricas (polígonos com espessura) que GIRAM sempre no mesmo sentido em
    torno de um pivô e/ou BATEM como prensa (vai até o centro rápido e volta devagar).
    Serve para: rotor de pás em cunha, moldura retangular, quadrado vazado, hastes em T, prensas.
    Cada peça: pts (coordenadas locais), closed, fill, off (pivô), w (rad/s), phase,
    press=dict(dir=(dx,dy), amp, period, ph) opcional, th (espessura do contorno)."""
    kind = "rods"

    def __init__(self, C, pieces, col=(185, 188, 195), hub=True):
        self.C = np.array(C, float); self.pieces = pieces; self.col = col; self.hub = hub
        self.th = 12.0
        self.p0 = self.C + np.array([0.0, -60.0])

    @staticmethod
    def press_d(pr, t):
        u = ((t / pr["period"]) + pr.get("ph", 0.0)) % 1.0
        if u < 0.22:
            x = (u / 0.22) ** 2
        elif u < 0.34:
            x = 1.0
        else:
            x = 1 - (u - 0.34) / 0.66
            x = x * x * (3 - 2 * x)
        return pr["amp"] * x

    def world(self, pc, t):
        w0 = pc.get("w", 0.0); k = pc.get("accel", 0.0)
        if k:
            a = pc.get("phase", 0.0) + w0 * (math.exp(k * t) - 1) / k; w0 = w0 * math.exp(k * t)
        else:
            a = pc.get("phase", 0.0) + w0 * t
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pv = self.C + np.array(pc.get("off", (0, 0)), float)
        tr = np.zeros(2); vt = np.zeros(2)
        if "press" in pc:
            pr = pc["press"]; dv = np.array(pr["dir"], float); dv /= np.linalg.norm(dv)
            d0 = self.press_d(pr, t); d1 = self.press_d(pr, t + 0.004)
            tr = dv * d0; vt = dv * (d1 - d0) / 0.004
        pts = pv + tr + (R @ np.array(pc["pts"], float).T).T
        return pts, pv + tr, vt, w0

    def segs(self, t):
        out = []
        for pc in self.pieces:
            pts, pv, vt, w = self.world(pc, t)
            out.append((pts[0], pts[1], pv, w if w else 1.0))
        return out

    def collide(self, pos, prev, mask, RB, t):
        hit_all = np.zeros(len(pos), bool)
        for pc in self.pieces:
            pts, pv, vt, w = self.world(pc, t)
            th = pc.get("th", self.th)
            lim = RB + th / 2
            n = len(pts); ns = n if pc.get("closed", True) else n - 1
            best = np.full(len(pos), 1e9); bcp = np.zeros_like(pos)
            for k in range(ns):
                a, b = pts[k], pts[(k + 1) % n]
                ab = b - a; L2 = ab @ ab + 1e-9
                s_ = np.clip(((pos - a) @ ab) / L2, 0, 1)
                cp = a + s_[:, None] * ab
                dd = np.linalg.norm(pos - cp, axis=1)
                m = dd < best
                best[m] = dd[m]; bcp[m] = cp[m]
            inside = np.zeros(len(pos), bool)
            if pc.get("fill") and pc.get("closed", True):
                x, y = pos[:, 0], pos[:, 1]
                for k in range(n):
                    a, b = pts[k], pts[(k + 1) % n]
                    c1 = (a[1] > y) != (b[1] > y)
                    xint = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-9) + a[0]
                    inside ^= c1 & (x < xint)
            hit = mask & ((best < lim) | inside)
            if hit.any():
                d = pos[hit] - bcp[hit]; dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4)
                nrm = d / dist[:, None]
                ins = inside[hit]
                nrm[ins] = -nrm[ins]
                pos[hit] = bcp[hit] + nrm * lim
                r = bcp[hit] - pv
                vs = w * np.stack([-r[:, 1], r[:, 0]], 1) + vt
                vv = (pos[hit] - prev[hit]) / DT
                vn = (vv * nrm).sum(1, keepdims=True)
                vv = vv + nrm * np.maximum(0, (vs * nrm).sum(1, keepdims=True) * 1.2 - vn)
                prev[hit] = pos[hit] - vv * DT
            hit_all |= mask & (best < lim + 2)
        return hit_all

    def draw(self, img, t, glowk=0.0):
        col = self.col
        dark = tuple(int(c * 0.45) for c in col); lite = tuple(min(255, c + 50) for c in col)
        for pc in self.pieces:
            pts, pv, vt, w = self.world(pc, t)
            th = pc.get("th", self.th)
            if "press" in pc:      # haste da prensa (vem da borda)
                pr = pc["press"]; dv = np.array(pr["dir"], float); dv /= np.linalg.norm(dv)
                base = self.C + np.array(pc.get("off", (0, 0)), float) - dv * 60
                ctr = pts.mean(0)
                cv2.line(img, P(base + np.array([6, 8])), P(ctr + np.array([6, 8])), (0, 0, 0), 22, cv2.LINE_AA, 4)
                cv2.line(img, P(base), P(ctr), dark, 22, cv2.LINE_AA, 4)
                cv2.line(img, P(base), P(ctr), col, 14, cv2.LINE_AA, 4)
            ip = (pts * SH).astype(np.int32)
            sh = ((pts + np.array([6, 8])) * SH).astype(np.int32)
            closed = pc.get("closed", True)
            if pc.get("fill"):
                cv2.fillPoly(img, [sh], (0, 0, 0), cv2.LINE_AA, 4)
                cv2.polylines(img, [sh], True, (0, 0, 0), int(th), cv2.LINE_AA, 4)
                cv2.polylines(img, [ip], True, dark, int(th + 3), cv2.LINE_AA, 4)
                cv2.fillPoly(img, [ip], col, cv2.LINE_AA, 4)
                cv2.polylines(img, [ip], True, col, int(th), cv2.LINE_AA, 4)
                cen = pts.mean(0)
                inner = ((cen + (pts - cen) * 0.72) * SH).astype(np.int32)
                cv2.polylines(img, [inner], True, lite, 2, cv2.LINE_AA, 4)
            else:
                cv2.polylines(img, [sh], closed, (0, 0, 0), int(th + 2), cv2.LINE_AA, 4)
                cv2.polylines(img, [ip], closed, dark, int(th + 4), cv2.LINE_AA, 4)
                cv2.polylines(img, [ip], closed, col, int(th), cv2.LINE_AA, 4)
                cv2.polylines(img, [ip], closed, lite, 2, cv2.LINE_AA, 4)
                if not closed:
                    for e in (pts[0], pts[-1]):
                        cv2.circle(img, P(e), int(th / 2 * SH), col, -1, cv2.LINE_AA, 4)
        if self.hub:
            cv2.circle(img, P(self.C + np.array([5, 7])), int(24 * SH), (0, 0, 0), -1, cv2.LINE_AA, 4)
            cv2.circle(img, P(self.C), int(24 * SH), dark, -1, cv2.LINE_AA, 4)
            cv2.circle(img, P(self.C), int(20 * SH), col, -1, cv2.LINE_AA, 4)
            cv2.circle(img, P(self.C), int(7 * SH), dark, -1, cv2.LINE_AA, 4)


class Bounce:
    """Sem mecanismo: a bolinha quica na borda e cada batida multiplica."""
    kind = "bounce"

    def __init__(self, C, **kw):
        self.C = np.array(C, float)

    def draw(self, img, t, glowk=0.0):
        pass


def P(p):
    return (int(round(p[0] * SH)), int(round(p[1] * SH)))


# ======================= simulação =======================
LEVELS = [1.012, 1.03, 1.05, 1.07]


def simulate(cfg, seed, keep=True):
    rng = np.random.default_rng(seed)
    ar = Arena(cfg["shape"], cfg.get("cy", 650))
    C = ar.C
    RB = cfg.get("RB", 9.0)
    pal = PALETTES[cfg.get("palette", "vivo")]
    if cfg["mech"] == "gears":
        mech = Gears(C, **cfg.get("gears", {}))
    elif cfg["mech"] == "rods":
        mech = Rods(C, **cfg["rods"])
    elif cfg["mech"] == "bounce":
        mech = Bounce(C)
    elif cfg["mech"] == "pieces":
        mech = Pieces(C, **cfg["pieces"])
    else:
        mech = Arcs(C, **cfg.get("arcs", {}))
    free = ar.area - (2 * math.pi * mech.Rc ** 2 if mech.kind == "gears" else 0)
    HEXMAX = free / (2 * math.sqrt(3) * RB * RB)
    CAP = int(HEXMAX * cfg.get("fill", 0.86))
    T_FULL = cfg.get("t_full", 33.0); GK = cfg.get("gk", 0.115)
    pos = np.zeros((0, 2)); prev = np.zeros((0, 2)); col = np.zeros(0, np.int32)
    trapped = np.zeros(0, bool); ghost = np.zeros(0, bool)
    frames, audio, pops = [], [], []
    st = dict(s=1.0, exploded=False, t_over=None, t_expl=None, lvl=0, t_lastpop=None, ci=0, rounds=0, final=None)
    frags, spawn_q, done = [], [], set()
    cdn = np.zeros(0)
    NF = int(DUR * FPS)

    def add(n, p0, v0, ci):
        nonlocal pos, prev, col, trapped, ghost
        base = math.atan2(v0[1], v0[0]); sp0 = np.linalg.norm(v0)
        ang = base + rng.normal(0, 0.55, n)
        rad = rng.uniform(0, 1, n) ** 0.5 * min(30, 4 + n * 0.08)
        p = p0 + np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
        sp = rng.uniform(0.6, 1.2, n) * sp0
        v = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
        pos = np.vstack([pos, p]); prev = np.vstack([prev, p - v * DT])
        col = np.concatenate([col, np.full(n, ci % len(pal), np.int32)])
        trapped = np.concatenate([trapped, np.zeros(n, bool)]); ghost = np.concatenate([ghost, np.zeros(n, bool)])
        try:
            cdn_ext(n)
        except NameError:
            pass

    def add_at(P_, V_, ci):
        nonlocal pos, prev, col, trapped, ghost, cdn
        n = len(P_)
        pos = np.vstack([pos, P_]); prev = np.vstack([prev, P_ - V_ * DT])
        col = np.concatenate([col, np.full(n, ci % len(pal), np.int32)])
        trapped = np.concatenate([trapped, np.zeros(n, bool)]); ghost = np.concatenate([ghost, np.zeros(n, bool)])
        cdn = np.concatenate([cdn, np.full(n, t_now[0])])

    t_now = [0.0]

    def add_at_cols(P_, V_, cols):
        nonlocal pos, prev, col, trapped, ghost, cdn
        n = len(P_)
        pos = np.vstack([pos, P_]); prev = np.vstack([prev, P_ - V_ * DT])
        col = np.concatenate([col, cols.astype(np.int32)])
        trapped = np.concatenate([trapped, np.zeros(n, bool)]); ghost = np.concatenate([ghost, np.zeros(n, bool)])
        cdn = np.concatenate([cdn, np.full(n, t_now[0])])

    def cdn_ext(n):
        nonlocal cdn
        cdn = np.concatenate([cdn, np.full(n, t_now[0])])

    # 1ª bolinha já no lugar de ser esmagada
    if mech.kind == "gears":
        p0 = mech.C + mech.n_in * (RB + 6)
    elif mech.kind == "bounce":
        p0 = C + np.array([0.0, -120.0])
    elif hasattr(mech, "p0"):
        p0 = mech.p0.copy()
    elif mech.kind == "rods":
        a0, b0, pv, w0 = mech.segs(0.6)[0]
        p0 = pv + (b0 - pv) * 0.6 + np.array([-(b0 - pv)[1], (b0 - pv)[0]]) / (np.linalg.norm(b0 - pv) + 1e-9) * (RB + mech.th / 2 + 1) * np.sign(w0)
    else:
        _, arcs0 = mech.arcs_at(mech.T1)
        p0 = arcs0[0][0].copy()
    add(1, p0, np.zeros(2) + 1e-6, 0); prev[:] = pos; ghost[:] = True
    for si in range(1, cfg.get("seeds", 1)):           # sementes de outras cores (manchas de cor)
        ang = 2 * math.pi * si / cfg["seeds"]
        q = C + np.array([math.cos(ang), math.sin(ang)]) * cfg.get("seed_r", 200)
        add(1, q, np.array([math.cos(ang + 1.6), math.sin(ang + 1.6)]) * 200, si if cfg.get("palette") == "manchas" else si * max(1, len(pal) // cfg["seeds"]))
        prev[-1] = pos[-1] - np.array([math.cos(ang + 1.6), math.sin(ang + 1.6)]) * 200 * DT
    t_tick, tick_i = 0.5, 0
    note_i = 0
    if mech.kind == "rods":
        ghost[:] = False
    if mech.kind == "bounce":
        ghost[:] = False; prev[:] = pos - np.array([[260.0, 0.0]]) * DT
    cdn = np.full(len(pos), -9.0)          # recarga de cada bolinha (bastões)
    acc = dict(n=0, p=np.zeros(2), k=0, t0=0.0)

    hits = np.zeros(len(pos), bool)
    for fi in range(NF):
        t = fi / FPS
        t_now[0] = t
        ex = st["exploded"]
        # relógio (tique-taque) — acelera na reta final
        while t_tick <= t:
            if not ex:
                audio.append((t_tick, "tick" if tick_i % 2 == 0 else "tock", 0.18)); tick_i += 1
            t_tick += (0.07 if st["lvl"] >= 3 else 0.5 / (1 + st["lvl"] * 1.3)) if (st["t_over"] is not None and not ex) else 0.5

        crushed = None
        if not ex:
            if mech.kind == "gears":
                k = int(t / 0.5)
                if ("bite", k) not in done and t >= 0.5 * k and k >= 1:
                    done.add(("bite", k))
                    sel = mech.pinched(pos, RB) & ~ghost
                    if k == 2:
                        sel |= ghost
                    if sel.any():
                        crushed = sel
                    if k % 4 == 0:                    # a cada 2 s muda a cor
                        st["ci"] += 1
                        st["rounds"] += 1
                        if st["t_over"] is not None:
                            st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                            audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                        audio.append((t, "trim", 0.55))
                    out_p, out_v = mech.out_pt, -mech.n_in * 260
            elif mech.kind in ("rods", "bounce"):
                k = int(t / 0.5)
                if ("rnd", k) not in done and k >= 1:
                    done.add(("rnd", k))
                    if k % 4 == 0:                    # a cada 2 s muda a cor (rodada)
                        st["ci"] += 1; st["rounds"] += 1
                        if st["t_over"] is not None:
                            st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                            audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                            pops.append((t, C.tolist(), 0, st["ci"] % len(pal)))
                        audio.append((t, "trim", 0.55))
                hb = np.where(hits & (t - cdn >= cfg.get("cool", 0.45)))[0]
                if len(hb):
                    target = CAP * math.exp(GK * (t - T_FULL))
                    limit = (HEXMAX * 0.97 * LEVELS[st["lvl"]] ** 2) if st["t_over"] is not None else CAP + 25
                    deficit = int(min(max(target, len(pos) + 1) , limit) - len(pos))
                    if st["t_over"] is not None:
                        deficit = int(limit - len(pos))
                    if deficit > 0:
                        per = int(np.clip(deficit / len(hb), 1, cfg.get("maxper", 3)))
                        rng.shuffle(hb)
                        nb = min(len(hb), int(math.ceil(deficit / per)))
                        src = hb[:nb]
                        cdn[src] = t
                        reps = np.repeat(src, per)[:deficit]
                        ang = rng.uniform(0, 2 * math.pi, len(reps))
                        off = np.stack([np.cos(ang), np.sin(ang)], 1) * RB * rng.uniform(0.9, 1.6, (len(reps), 1))
                        vel = (pos[reps] - prev[reps]) / DT * 0.7 + off * (30 if mech.kind == "bounce" else 9)
                        if cfg.get("inherit"):
                            add_at_cols(pos[reps] + off, vel, col[reps])
                        else:
                            add_at(pos[reps] + off, vel, st["ci"])
                        acc["n"] += len(reps); acc["p"] = acc["p"] + pos[src].sum(0); acc["k"] += len(src)
                hits = np.zeros(len(pos), bool)
                if acc["n"] and t - acc["t0"] >= 0.2:
                    gc = acc["p"] / max(1, acc["k"])
                    pops.append((t, gc.tolist(), acc["n"], st["ci"] % len(pal)))
                    audio.append((t, "tom%d" % (note_i % 8), 0.5)); note_i += 1
                    if acc["n"] >= 6:
                        audio.append((t, "plim", 0.35))
                    acc.update(n=0, p=np.zeros(2), k=0, t0=t)
            else:
                kk = int((t - mech.T1) // mech.T) if t >= mech.T1 else -1
                ta = mech.T1 + kk * mech.T
                if kk >= 0 and ("al", kk) not in done:
                    done.add(("al", kk))
                    _, arcs = mech.arcs_at(ta)
                    dd = np.linalg.norm(pos - arcs[0][0], axis=1)
                    trapped |= dd < mech.RA - mech.TH / 2 - RB * 0.3
                    ghost[:] = False
                    audio.append((ta, "clink", 0.35))
                if kk >= 0 and ("pop", kk) not in done and t >= ta + 0.72:
                    done.add(("pop", kk))
                    if trapped.any():
                        crushed = trapped.copy()
                    st["ci"] += 1; st["rounds"] += 1
                    if st["t_over"] is not None:
                        st["lvl"] = min(3, st["lvl"] + 1); st["t_lastpop"] = t
                        audio.append((t, "rumble", 0.35 + 0.15 * st["lvl"]))
                    m, arcs = mech.arcs_at(t)
                    out_p = C + 0.45 * ((arcs[0][0] + arcs[1][0]) - 2 * C)
                    out_v = np.array([math.cos(m), math.sin(m)]) * 250
            if crushed is not None:
                nt = int(crushed.sum())
                km = ~crushed
                pos, prev, col, trapped, ghost, cdn = pos[km], prev[km], col[km], trapped[km], ghost[km], cdn[km]
                target = CAP * math.exp(GK * (t + 0.5 - T_FULL))
                n = nt * 2 + int(max(0.0, target - len(pos)) * (0.35 if mech.kind == "gears" else 1.0))
                if st["t_over"] is not None:
                    capn = int(HEXMAX * 0.95 * LEVELS[st["lvl"]] ** 2)
                    n = max(nt, min(n, capn - len(pos)))
                else:
                    n = max(nt, min(n, CAP + 20 - len(pos)))
                pulses = max(1, min(8, (n + 2) // 3)); base, rest = divmod(n, pulses)
                for p in range(pulses):
                    spawn_q.append([t + 0.04 + p * 0.06, base + (1 if p < rest else 0), out_p.copy(), out_v.copy(), st["ci"]])
                pops.append((t, out_p.tolist(), n, st["ci"] % len(pal)))
                audio.append((t, "plim", 0.6))
        spawn_q.sort(key=lambda x: x[0])
        while spawn_q and spawn_q[0][0] <= t:
            ts, n, p0, v0, ci = spawn_q.pop(0)
            if not st["exploded"] and n > 0:
                add(n, p0, v0, ci)
                audio.append((ts, "tom%d" % (note_i % 8), 0.5)); note_i += 1

        if not ex and st["t_over"] is None and len(pos) >= CAP:
            st["t_over"] = t; audio.append((t, "rumble", 0.5))
        if not ex and st["t_over"] is not None:
            st["s"] += (LEVELS[st["lvl"]] - st["s"]) * 0.06
            if st["lvl"] >= 3 and t >= st["t_lastpop"] + 1.0:
                st.update(exploded=True, t_expl=t, final=len(pos)); ex = True
                audio.append((t - 1.0, "riser", 0.8)); audio.append((t, "boom", 1.0)); audio.append((t, "sparkle", 0.9))
                d = pos - C
                dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-3
                spd = rng.uniform(700, 1500, (len(pos), 1)) * (0.4 + 0.6 * dist / 300)
                prev = pos - (d / dist * spd + rng.normal(0, 120, pos.shape)) * DT
                Pp = ar.dense(st["s"])
                step = max(1, len(Pp) // 40)
                for i in range(0, len(Pp), step):
                    a, b = Pp[i], Pp[(i + step) % len(Pp)]
                    mm = (a + b) / 2; dv = (mm - C) / (np.linalg.norm(mm - C) + 1e-6)
                    frags.append(dict(a=a - mm, b=(b - mm) * 0.9, c=mm, rot=0.0, v=dv * rng.uniform(500, 1100), w=rng.uniform(-6, 6)))

        grav = np.array([0.0, 900.0]) if (ex or (mech.kind == "bounce" and st["t_over"] is None)) else np.zeros(2)
        for s in range(SUB):
            if len(pos) == 0:
                break
            ts = t + s * DT
            v = (pos - prev) / DT * 0.9992
            if not ex:
                if mech.kind == "gears":
                    mech.pull(pos, v, RB)
                elif mech.kind in ("rods", "bounce"):
                    v *= 0.999
                else:
                    rel = pos - C
                    v += (np.stack([-rel[:, 1], rel[:, 0]], 1) * -0.25 - v) * min(1.0, 0.4 * DT)
                v[ghost] = 0
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            cap = 2500 if ex else 650
            v = np.where(spd > cap, v / np.maximum(spd, 1e-6) * cap, v)
            prev = pos; pos = pos + v * DT + grav * DT * DT
            for it in range(2):
                pr = cKDTree(pos).query_pairs(2 * RB, output_type="ndarray")
                if len(pr):
                    i, j = pr[:, 0], pr[:, 1]
                    d = pos[j] - pos[i]
                    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-4)
                    corr = d * ((2 * RB - dist) / dist * 0.5 * 0.85)[:, None]
                    dp = np.zeros_like(pos); np.add.at(dp, i, -corr); np.add.at(dp, j, corr)
                    pos += dp
                if not ex:
                    solid = ~ghost
                    if mech.kind == "gears":
                        mech.collide(pos, prev, solid, RB)
                    elif mech.kind == "rods":
                        if len(hits) != len(pos):
                            hits = np.zeros(len(pos), bool)
                        hits |= mech.collide(pos, prev, solid, RB, ts)
                    elif mech.kind == "bounce":
                        pass
                    else:
                        _, arcs = mech.arcs_at(ts)
                        free = solid & ~trapped
                        for idx, (c, a0, a1) in enumerate(arcs):
                            d = pos - c
                            dist = np.linalg.norm(d, axis=1)
                            relA = (np.arctan2(d[:, 1], d[:, 0]) - a0) % (2 * math.pi)
                            hit = free & (relA <= mech.SPAN) & (np.abs(dist - mech.RA) < RB + mech.TH / 2)
                            if hit.any():
                                sg = np.sign(dist[hit] - mech.RA); sg[sg == 0] = 1
                                pos[hit] = c + d[hit] / np.maximum(dist[hit], 1e-4)[:, None] * (mech.RA + sg * (RB + mech.TH / 2))[:, None]
                            if trapped.any():
                                d = pos - c; dist = np.linalg.norm(d, axis=1)
                                lim = mech.RA - mech.TH / 2 - RB
                                o = trapped & (dist > lim)
                                if o.any():
                                    pos[o] = c + d[o] / dist[o][:, None] * lim
                        dg = pos - C; dd = np.linalg.norm(dg, axis=1)
                        hg = solid & (dd < RB + mech.GEAR)
                        if hg.any():
                            pos[hg] = C + dg[hg] / np.maximum(dd[hg], 1e-4)[:, None] * (RB + mech.GEAR)
                    sd, gx, gy = ar.sample(pos, st["s"])
                    o = sd < RB
                    if o.any():
                        g = np.stack([gx[o], gy[o]], 1)
                        if mech.kind == "bounce":
                            if len(hits) != len(pos):
                                hits = np.zeros(len(pos), bool)
                            vn0 = ((pos[o] - prev[o]) * g).sum(1)
                            oi = np.where(o)[0]
                            hits[oi[vn0 < -0.3]] = True
                        pos[o] += g * (RB - sd[o])[:, None]
                        vv = pos[o] - prev[o]; vn = (vv * g).sum(1)
                        prev[o] = pos[o] - (vv - np.minimum(vn, 0)[:, None] * g * (1.97 if mech.kind == "bounce" else 1.85))
        if ex:
            kp = (pos[:, 0] > -60) & (pos[:, 0] < W + 60) & (pos[:, 1] < H + 60) & (pos[:, 1] > -600)
            pos, prev, col, trapped, ghost, cdn = pos[kp], prev[kp], col[kp], trapped[kp], ghost[kp], cdn[kp]
            for fr in frags:
                fr["c"] = fr["c"] + fr["v"] / FPS; fr["v"] = fr["v"] + np.array([0, 900.0]) / FPS; fr["rot"] += fr["w"] / FPS
        if keep:
            frames.append(dict(t=t, pos=pos.copy(), col=col.copy(), s=st["s"], exploded=ex, t_over=st["t_over"],
                               lvl=st["lvl"], t_lastpop=st["t_lastpop"], t_expl=st["t_expl"], n=len(pos) if not ex else st["final"],
                               frags=[dict(f) for f in frags], ci=st["ci"]))
        if ex and t > st["t_expl"] + 4.0:
            break
    return frames, audio, pops, st, ar, mech, CAP


# ======================= desenho =======================
def make_ball_sprite(color, r, ss=6):
    s = int(math.ceil(r)) * 2 + 4
    S = s * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / ss - s / 2 + 0.5 / ss
    d = np.sqrt(xx ** 2 + yy ** 2)
    a = np.clip((r - d) + 0.5, 0, 1)
    c = np.array(color, np.float32)
    nz = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    shade = 0.55 + 0.55 * np.clip((xx / r) * -0.45 + (yy / r) * -0.55 + nz * 0.7, 0, 1)
    rgb = c[None, None] * shade[..., None]
    rgb = rgb * (1 - 0.45 * np.clip((d / r - 0.82) / 0.18, 0, 1)[..., None])
    hl = np.clip(1 - np.sqrt((xx / r + 0.38) ** 2 + (yy / r + 0.42) ** 2) / 0.32, 0, 1) ** 1.6
    rgb = np.clip(rgb + (255 - rgb) * hl[..., None] * 0.9, 0, 255)
    small = cv2.resize(np.dstack([rgb, a * 255]).astype(np.float32), (s, s), interpolation=cv2.INTER_AREA)
    al = small[..., 3:4] / 255
    return (small[..., :3] * al).astype(np.float32), (1 - al).astype(np.float32)


def to_pre(im):
    a = np.asarray(im).astype(np.float32); al = a[..., 3:4] / 255
    return (a[..., [2, 1, 0]] * al).astype(np.float32), (1 - al).astype(np.float32)


_tc = {}


def put_text(f, txt, size, cx, cy, color=(255, 255, 255), stroke=0, stroke_col=(0, 0, 0), alpha=1.0, scale=1.0):
    key = (txt, size, color, stroke, stroke_col)
    if key not in _tc:
        font = ImageFont.truetype(FONT_B, size)
        bb = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), txt, font=font, stroke_width=stroke)
        im = Image.new("RGBA", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), txt, font=font, fill=(color[2], color[1], color[0], 255),
                                stroke_width=stroke, stroke_fill=(stroke_col[2], stroke_col[1], stroke_col[0], 255))
        _tc[key] = to_pre(im)
    pre, inv = _tc[key]
    if abs(scale - 1) > 0.02:
        h, w = pre.shape[:2]; nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pre = cv2.resize(pre, (nw, nh)); inv = cv2.resize(inv[..., 0], (nw, nh))[..., None]
    blit(f, pre, inv, int(cx - pre.shape[1] / 2), int(cy - pre.shape[0] / 2), alpha)


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


def emoji(ch, d):
    key = ("emo", ch, d)
    if key not in _tc:
        f = ImageFont.truetype(EMOJI_FONT, 109)
        im = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((12, 12), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox()).resize((d, d), Image.LANCZOS)
        _tc[key] = to_pre(im)
    return _tc[key]


def ease_out_back(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Renderer:
    def __init__(self, cfg, ar, mech, RB):
        self.cfg, self.ar, self.mech = cfg, ar, mech
        self.pal = PALETTES[cfg.get("palette", "vivo")]
        self.SPR = [make_ball_sprite(c, RB) for c in self.pal]
        self.SS = self.SPR[0][0].shape[0]; self.HALF = self.SS // 2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = np.clip(1 - np.sqrt((xx - ar.C[0]) ** 2 + (yy - ar.C[1]) ** 2) / 760, 0, 1) ** 1.5
        bg = np.zeros((H, W, 3), np.float32); bg[:] = (8, 6, 8)
        if cfg.get("theme") != "minimal":
            bg += g[..., None] * np.array(cfg.get("bg", (40, 22, 30)), np.float32)
        else:
            bg[:] = cfg.get("bg_min", (6, 6, 6))
            if cfg.get("grid"):
                bg[::40, :] += 7; bg[:, ::40] += 7
        fl = cv2.GaussianBlur((ar.mask > 0).astype(np.float32), (0, 0), 1.0)[..., None]
        self.BG = bg; self.FL = fl
        self.GL = self.glow(150)

    @staticmethod
    def glow(r):
        s = int(r * 2); yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - s / 2
        return np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 2

    def add_glow(self, f, c, color, k):
        g = self.GL; s = g.shape[0]
        x0, y0 = int(c[0] - s / 2), int(c[1] - s / 2)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
        if xa < xb and ya < yb and k > 0.01:
            f[ya:yb, xa:xb] += g[ya - y0:yb - y0, xa - x0:xb - x0][..., None] * k * (np.array(color, np.float32) * 0.8 + 50)

    def render(self, fr, pops, st, srng, CAP):
        ar, mech = self.ar, self.mech
        t = fr["t"]; ex = fr["exploded"]
        f = self.BG.copy()
        if not ex:
            s = fr["s"]
            fl = self.FL if abs(s - 1) < 0.002 else cv2.warpAffine(self.FL, np.float32([[s, 0, ar.C[0] * (1 - s)], [0, s, ar.C[1] * (1 - s)]]), (W, H))[..., None]
            if self.cfg.get("theme") != "minimal":
                f = f * (1 - fl * 0.5) + fl * 0.5 * np.array(self.cfg.get("floor", (30, 22, 26)), np.float32)
        for (t0, gc, n, ci) in (pops if self.cfg.get("theme") != "minimal" else []):
            if t0 - 0.02 <= t < t0 + 0.8:
                k = 1.0 if t < t0 + 0.1 else max(0.0, 1 - (t - t0 - 0.1) / 0.7)
                self.add_glow(f, gc, self.pal[ci], k * 0.9)
        ip = np.round(fr["pos"]).astype(np.int32); cl = fr["col"]
        SS, HALF = self.SS, self.HALF
        for k in range(len(ip)):
            pre, inv = self.SPR[cl[k]]
            x0, y0 = ip[k, 0] - HALF, ip[k, 1] - HALF
            if 0 <= x0 and 0 <= y0 and x0 + SS <= W and y0 + SS <= H:
                reg = f[y0:y0 + SS, x0:x0 + SS]; reg *= inv; reg += pre
            else:
                blit(f, pre, inv, x0, y0)
        img = np.clip(f, 0, 255).astype(np.uint8)
        wob = 0.0
        if fr["t_over"] is not None and not ex:
            lv = fr["lvl"]
            kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.6) if fr["t_lastpop"] is not None else 0.0
            kick0 = max(0.0, 1 - (t - fr["t_over"]) / 0.8)
            wob = [3, 6, 10, 15][lv] + 10 * kick + 8 * kick0
            if lv >= 3:
                wob += 10 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
        cur = self.pal[fr["ci"] % len(self.pal)]
        if not ex:
            pts = (ar.dense(fr["s"], wob, t) * SH).astype(np.int32)
            if self.cfg.get("theme") != "minimal":
                gl = np.zeros_like(img)
                cv2.polylines(gl, [pts], True, cur, 14, cv2.LINE_AA, 4)
                img = cv2.add(img, (cv2.GaussianBlur(gl, (0, 0), 9) * 0.8).astype(np.uint8))
            cv2.polylines(img, [pts], True, (245, 240, 250), 4 if self.cfg.get("theme") != "minimal" else 3, cv2.LINE_AA, 4)
        for fg in fr["frags"]:
            cr, sr = math.cos(fg["rot"]), math.sin(fg["rot"]); R = np.array([[cr, -sr], [sr, cr]])
            cv2.line(img, P(fg["c"] + R @ fg["a"]), P(fg["c"] + R @ fg["b"]), (245, 240, 250), 4, cv2.LINE_AA, 4)
        ta = 1.0 if not ex else max(0.0, 1 - (t - fr["t_expl"]) / 0.25)
        if ta > 0:
            ov = img.copy(); mech.draw(ov, t)
            img = ov if ta >= 1 else cv2.addWeighted(ov, ta, img, 1 - ta, 0)
        f = img.astype(np.float32)
        # "+N" de cada leva
        for (t0, gc, n, ci) in pops:
            dtt = t - t0
            if 0 <= dtt < 0.8 and not ex and n > 0 and self.cfg.get("theme") != "minimal":
                put_text(f, f"+{n}", 26, gc[0], gc[1] + 20 + dtt * 50, self.pal[ci], stroke=4, stroke_col=(20, 15, 20), alpha=min(1.0, (1 - dtt / 0.8) * 1.6))
        # gancho + contador
        top = ar.top
        if not ex:
            hk = self.cfg.get("hook", "WATCH IT MULTIPLY!"); hs = 42
            while hs > 24 and ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(hk, font=ImageFont.truetype(FONT_B, hs)) > 670:
                hs -= 2
            put_text(f, hk, hs, W / 2, 84, (255, 255, 255), stroke=4, stroke_col=(30, 20, 40))
            if self.cfg.get("hook_pt"):
                put_text(f, self.cfg["hook_pt"], 24, W / 2, 124, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40))
            blit(f, *emoji("\U0001F447", 34), int(W / 2 - 17), int(146 + 4 * math.sin(t * 6)))
            mn = self.cfg.get("theme") == "minimal"
            ctxt = str(fr['n']) if self.cfg.get("num_only") else (f"{fr['n']} BALLS" if fr['n'] != 1 else "1 BALL")
            if self.cfg.get("rpm") is not None and hasattr(self.mech, "world"):
                _, _, _, wv = self.mech.world(self.mech.pieces[self.cfg["rpm"]], t)
                ctxt = f"{abs(wv) * 60 / (2 * math.pi):.0f} RPM"
            put_text(f, ctxt, (64 if self.cfg.get("num_only") else 40) if mn else 54, W / 2, max(205, top - 75),
                     (235, 235, 235) if mn else cur, stroke=0 if mn else 5, stroke_col=(20, 15, 20))
            if fr["t_over"] is not None:
                put_text(f, "ABOUT TO EXPLODE!", 50, W / 2, ar.bottom + 70, (60, 60, 255), stroke=5, stroke_col=(255, 255, 255), scale=1 + 0.08 * math.sin(t * 18))
                put_text(f, "Vai explodir!", 26, W / 2, ar.bottom + 116, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40))
            else:
                put_text(f, "Follow for more!", 30, W / 2, ar.bottom + 62, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40))
                put_text(f, "Segue a página pra ver mais", 22, W / 2, ar.bottom + 96, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40))
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
                put_text(f, f"{st['final']} BALLS!", 70, W / 2, 680, cur, stroke=6, stroke_col=(255, 255, 255), scale=max(0.05, sc))
                put_text(f, f"{st['final']} bolinhas!", 30, W / 2, 742, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40), scale=max(0.05, sc))
                if te > 1.3:
                    put_text(f, "Liked it? FOLLOW US!", 44, W / 2, 805, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40), alpha=min(1, (te - 1.3) / 0.3))
                    put_text(f, "Gostou? Segue a página!", 24, W / 2, 846, (200, 205, 215), stroke=3, stroke_col=(30, 20, 40), alpha=min(1, (te - 1.3) / 0.3))
                    put_text(f, "Share it with a friend", 32, W / 2, 900, (200, 230, 255), stroke=3, stroke_col=(30, 20, 40), alpha=min(1, (te - 1.3) / 0.3))
                    put_text(f, "Compartilha com alguém", 22, W / 2, 936, (255, 255, 255), stroke=3, stroke_col=(30, 20, 40), alpha=min(1, (te - 1.6) / 0.3) if te > 1.6 else 0.0)
                img = np.clip(f, 0, 255).astype(np.uint8)
            shake = max(0.0, 18 * (1 - te / 0.7))
        elif fr["t_over"] is not None:
            lv = fr["lvl"]
            kick = max(0.0, 1 - (t - fr["t_lastpop"]) / 0.5) if fr["t_lastpop"] is not None else 0.0
            shake = [1.0, 2.5, 4.0, 6.0][lv] + 7 * kick + 4 * max(0.0, 1 - (t - fr["t_over"]) / 0.6)
            if lv >= 3:
                shake += 6 * min(1.0, (t - fr["t_lastpop"]) / 1.0) ** 2
        else:
            shake = 0
        if shake > 0.5:
            img = cv2.warpAffine(img, np.float32([[1, 0, srng.normal() * shake], [0, 1, srng.normal() * shake]]), (W, H))
        return img


# ======================= áudio =======================
SR = 44100


def make_audio(events, dur, path, scale_name="maior", seed=3):
    rng = np.random.default_rng(seed)
    N = int(SR * dur); out = np.zeros((N, 2), np.float32)
    def env(n, tau): return np.exp(-np.arange(n) / SR / tau)
    def lp(x, a): return lfilter([a], [1, a - 1], x)
    def tt(n): return np.arange(n) / SR
    def tone(f, du, tau, harm=((1, 1.0),)):
        n = int(SR * du); t = tt(n)
        return sum(a * np.sin(2 * np.pi * f * h * t) for h, a in harm) * env(n, tau) * np.minimum(1, t / 0.002)
    def marimba(f):   # "tom"
        return tone(f, 0.45, 0.12, ((1, 1), (3.98, 0.25), (9.2, 0.06)))
    scales = {"maior": [261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3],
              "menor": [261.6, 311.1, 349.2, 392.0, 466.2, 523.3, 622.3, 698.5]}
    sc = scales[scale_name]
    melody = [0, 2, 4, 5, 4, 2, 3, 1, 0, 4, 6, 7, 5, 3, 2, 4]
    def click(f1, f2, tau):
        n = int(SR * 0.08); t = tt(n)
        return (0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)) * env(n, tau) + rng.normal(0, 1, n) * env(n, 0.0015) * 0.4
    def trim():
        s = np.zeros(int(SR * 0.45))
        for i, f in enumerate((1568.0, 1760.0, 2093.0)):
            x = tone(f, 0.3, 0.08, ((1, 1), (2, 0.3)))
            i0 = int(i * 0.05 * SR); s[i0:i0 + len(x)] += x[: len(s) - i0]
        return s
    def plim():
        return tone(2637.0, 0.7, 0.22, ((1, 1), (2.01, 0.35), (3.0, 0.12))) + tone(1318.5, 0.7, 0.3, ((1, 0.5),))
    def clink():
        n = int(SR * 0.25); t = tt(n)
        return (np.sin(2 * np.pi * 2350 * t) + 0.6 * np.sin(2 * np.pi * 3720 * t)) * env(n, 0.05)
    def rumble():
        n = int(SR * 2.0); t = tt(n); e = (t / 2.0) ** 2
        return lp(rng.normal(0, 1, n), 0.01) * 8 * e + np.sin(2 * np.pi * 55 * t) * e * 0.4
    def riser():
        n = int(SR * 1.0); t = tt(n)
        nz = rng.normal(0, 1, n)
        sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * (t / 1.0) ** 2) / SR) * 0.4
        return (lfilter([0.2], [1, -0.8], nz) * 1.5 + sweep) * (t / 1.0) ** 2
    def boom():
        n = int(SR * 2.6); t = tt(n); nz = rng.normal(0, 1, n)
        return lp(nz, 0.02) * 6 * env(n, 0.55) + lp(nz, 0.35) * env(n, 0.09) * 1.2 + \
            np.sin(2 * np.pi * np.cumsum(75 - 45 * np.minimum(t, 1)) / SR) * env(n, 0.6) * 1.1
    def sparkle():
        s = np.zeros(int(SR * 2.8))
        notes = [2093.0, 2349.3, 2637.0, 3136.0, 3520.0, 4186.0]
        for i in range(36):
            x = tone(notes[rng.integers(0, 6)] * rng.choice([1, 0.5]), 0.5, 0.12, ((1, 1), (2.76, 0.3)))
            i0 = int((0.15 + rng.uniform(0, 1.8) ** 1.2) * SR)
            if i0 < len(s):
                s[i0:i0 + len(x)] += x[: len(s) - i0] * rng.uniform(0.3, 0.9)
        return s
    base = {"tick": click(3100, 4700, 0.006), "tock": click(1750, 2650, 0.009), "trim": trim(), "plim": plim(),
            "clink": clink(), "rumble": rumble(), "riser": riser(), "boom": boom(), "sparkle": sparkle()}
    for i in range(8):
        base["tom%d" % i] = marimba(sc[melody[(i * 3) % len(melody)] % len(sc)])
    melody_i = 0
    base = {k: v / np.abs(v).max() for k, v in base.items()}
    gain = {"boom": 2.6, "sparkle": 1.5, "riser": 1.2, "rumble": 1.3, "plim": 0.9, "trim": 0.8}
    for (t, kind, vol) in events:
        if kind.startswith("tom"):
            s = marimba(sc[melody[melody_i % len(melody)]]); s = s / np.abs(s).max(); melody_i += 1
            s = s * vol * 0.55
        else:
            s = base[kind] * vol * gain.get(kind, 1.0)
        i0 = int(t * SR)
        if i0 >= N or i0 < 0:
            continue
        seg = s[: N - i0]
        pan = {"tick": 0.2, "tock": -0.2}.get(kind, 0.0)
        out[i0:i0 + len(seg), 0] += seg * (1 - pan); out[i0:i0 + len(seg), 1] += seg * (1 + pan)
    # ---- camada "drive" (estudo das referências: som presente desde o 1º segundo, ritmo que
    # ACELERA até o fim, densidade de estalos crescendo junto com as bolinhas, pausa antes do boom)
    tb = [t for (t, k, v) in events if k == "boom"]
    t_boom = tb[0] if tb else dur - 3
    drv = np.zeros(N, np.float32)
    def put(x, t0, g):
        i0 = int(t0 * SR)
        if 0 <= i0 < N:
            seg = x[: N - i0]; drv[i0:i0 + len(seg)] += seg * g
    nk = int(SR * 0.22); tk = tt(nk)
    kick = np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-tk / 0.03)) / SR) * env(nk, 0.11)
    nh = int(SR * 0.05); hat = lfilter([1, -1], [1, -0.3], rng.normal(0, 1, nh)) * env(nh, 0.008) * 0.25
    # gancho sonoro no 1º segundo: "whoosh" + impacto
    nw = int(SR * 0.5); tw = tt(nw)
    whoosh = lfilter([0.15], [1, -0.85], rng.normal(0, 1, nw)) * (tw / 0.5) ** 2 * 1.6
    put(whoosh, 0.0, 0.5); put(kick * 1.3, 0.5, 0.9); put(base["plim"], 0.5, 0.35)
    t_end = max(4.0, t_boom - 1.05)
    t = 0.5
    while t < t_end:
        fr = t / t_end
        bpm = 96 + 84 * fr ** 1.4
        beat = 60.0 / bpm
        put(kick, t, 0.30 + 0.35 * fr)
        put(hat, t + beat / 2, 0.25 + 0.45 * fr)
        if fr > 0.55:
            put(hat, t + beat / 4, 0.18 * fr); put(hat, t + 3 * beat / 4, 0.18 * fr)
        t += beat
    # estalos de bolinhas (marble crackle) com densidade crescente
    nc = int(SR * 0.012); tc_ = tt(nc)
    t = 0.6
    while t < t_boom - 0.3:
        rate = min(45.0, 1.5 * math.exp(t / 8.5))
        f0 = rng.uniform(2200, 5200)
        put(np.sin(2 * np.pi * f0 * tc_) * env(nc, 0.0025), t, rng.uniform(0.04, 0.10))
        t += rng.exponential(1.0 / rate)
    # respiro: silencia o "drive" 0,25 s antes do boom (tensão), o riser segue
    i0 = int((t_boom - 0.25) * SR)
    if 0 < i0 < N:
        drv[i0:] = 0
    out[:, 0] += drv; out[:, 1] += drv
    for d, g in ((0.05, 0.22), (0.12, 0.13), (0.21, 0.07)):
        k = int(d * SR); out[k:] += out[:-k].copy() * g
    out /= np.abs(out).max() / 0.9
    w = wave.open(path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes()); w.close()


# ======================= CLI =======================
CONFIGS = {
    # ----- 10/11 (refeitos a partir das referências da pasta do dono) -----
    "acelera_martelo": dict(shape="circle", mech="pieces", palette="vivo", theme="minimal", seeds=1, RB=6.5, maxper=4, rpm=0,
                            hook="EVERY SECOND IT SPINS FASTER", hook_pt="A cada segundo gira mais rápido",
                            pieces=dict(col=(190, 194, 200), pieces=[
                                dict(pts=[(-262, 0), (262, 0)], closed=False, th=12, w=1.05, accel=0.046),
                                dict(pts=[(205, -34), (205, 34)], closed=False, th=14, w=1.05, accel=0.046),
                                dict(pts=[(-205, -34), (-205, 34)], closed=False, th=14, w=1.05, accel=0.046)]),
                            scale="maior", t_full=30.0),
    "colchetes_trilho": dict(shape="circle", mech="pieces", palette="neon", theme="minimal", seeds=3, RB=6.5, maxper=4,
                             hook="THE CLAMPS THAT MULTIPLY", hook_pt="As garras que multiplicam",
                             pieces=dict(hub=False, col=(175, 180, 188), pieces=[
                                 dict(off=(-170, 0), pts=[(10, -72), (-30, -72), (-30, 72), (10, 72)], closed=False, th=14, press=dict(dir=(1, 0), amp=130, period=1.6, ph=0.0)),
                                 dict(off=(170, 0), pts=[(-10, -72), (30, -72), (30, 72), (-10, 72)], closed=False, th=14, press=dict(dir=(-1, 0), amp=130, period=1.6, ph=0.0)),
                                 dict(pts=[(-300, 0), (300, 0)], closed=False, th=8)]),
                             scale="menor", t_full=30.0),
    "quadrado_arcos": dict(shape="square", mech="pieces", palette="manchas", theme="minimal", inherit=True, seeds=5, RB=7.0, seed_r=150,
                           hook="4 ARCS SWINGING INTO THE BOX", hook_pt="4 arcos entrando na caixa",
                           pieces=dict(hub=False, col=(170, 174, 182), pieces=[
                               dict(off=(sx * 310, sy * 310), pts=[(150 * math.cos(a), 150 * math.sin(a)) for a in np.linspace(0, math.pi, 14)],
                                    closed=False, th=13, w=0.9 * sx * sy, phase=ph)
                               for (sx, sy, ph) in ((-1, -1, 0.0), (1, -1, 1.6), (1, 1, 3.1), (-1, 1, 4.7))]),
                           scale="maior", t_full=30.0),
    "octogono_x_setas": dict(shape="octagon", mech="pieces", palette="vivo", theme="minimal", seeds=3, RB=6.5, maxper=4,
                             hook="AN X IN THE MIDDLE, ARROWS ON THE SIDES", hook_pt="Um X no meio e setas dos lados",
                             pieces=dict(col=(178, 182, 190), pieces=[
                                 dict(pts=[(-120, 0), (120, 0)], closed=False, th=13, w=1.25, phase=math.pi / 4),
                                 dict(pts=[(-120, 0), (120, 0)], closed=False, th=13, w=1.25, phase=-math.pi / 4),
                                 dict(off=(-300, 0), pts=[(-40, -75), (35, 0), (-40, 75)], closed=False, th=14, press=dict(dir=(1, 0), amp=120, period=1.8, ph=0.0)),
                                 dict(off=(300, 0), pts=[(40, -75), (-35, 0), (40, 75)], closed=False, th=14, press=dict(dir=(-1, 0), amp=120, period=1.8, ph=0.5))]),
                             scale="menor", t_full=30.0),
    "jogo_da_velha": dict(shape="circle", mech="pieces", palette="vivo", theme="minimal", inherit=True, seeds=9, seed_r=215, RB=7.0, fill=0.9,
                          hook="A GIANT # SPINNING", hook_pt="Um # gigante girando",
                          pieces=dict(hub=False, col=(196, 199, 206), pieces=[
                              dict(pts=[(-330, -105), (330, -105)], closed=False, th=12, w=0.7, phase=math.pi / 4),
                              dict(pts=[(-330, 105), (330, 105)], closed=False, th=12, w=0.7, phase=math.pi / 4),
                              dict(pts=[(-105, -330), (-105, 330)], closed=False, th=12, w=0.7, phase=math.pi / 4),
                              dict(pts=[(105, -330), (105, 330)], closed=False, th=12, w=0.7, phase=math.pi / 4)]),
                          scale="maior", t_full=30.0),
    "roda_e_martelo": dict(shape="circle", mech="pieces", palette="manchas", theme="minimal", inherit=True, seeds=5, RB=7.0, fill=0.92, maxper=4,
                           hook="THE WHEEL VS THE HAMMER", hook_pt="A roda contra o martelo",
                           pieces=dict(col=(200, 210, 120), pieces=[
                               dict(pts=[(32, 0), (268, 0)], closed=False, th=8, w=0.45, phase=k * 2 * math.pi / 5) for k in range(5)] + [
                               dict(pts=[(0, 0), (238, 0)], closed=False, th=13, w=-1.1),
                               dict(pts=[(238, -36), (238, 36)], closed=False, th=16, w=-1.1)]),
                           scale="menor", t_full=30.0),
    "minimal_roda_raios": dict(shape="circle", mech="rods", palette="manchas", theme="minimal", inherit=True, seeds=4, hook="THE WHEEL THAT MULTIPLIES", hook_pt="A roda que multiplica",
                               rods=dict(rods=[dict(off=(0, 0), L=250, w=1.3, phase=k * 2 * math.pi / 5, half=True) for k in range(5)],
                                         col=(200, 205, 210), th=10.0), scale="maior", t_full=30.0),
    "circulo_quica": dict(shape="circle", mech="bounce", palette="vivo", hook="EVERY BOUNCE = +1 BALL", hook_pt="Cada batida = +1 bolinha", bg=(25, 35, 60),
                          scale="maior", t_full=30.0, cool=0.6, maxper=2),
    "relogio_ponteiro": dict(shape="circle", mech="rods", palette="neon", hook="THE CLOCK THAT MULTIPLIES BALLS", hook_pt="O relógio que multiplica bolinhas", bg=(40, 30, 20),
                             rods=dict(rods=[dict(off=(0, 0), L=268, w=1.6, phase=-math.pi / 2, half=True)], col=(230, 230, 240), th=14.0, marks=298),
                             scale="menor", t_full=30.0),
    "quadrado_4bastoes": dict(shape="square", mech="rods", palette="vivo", hook="4 SPINNING BARS MULTIPLYING BALLS", hook_pt="4 bastões multiplicando bolinhas", bg=(40, 22, 30),
                              rods=dict(rods=[dict(off=(-150, -150), L=112, w=2.1, phase=0.0), dict(off=(150, -150), L=112, w=-2.1, phase=0.0),
                                              dict(off=(150, 150), L=112, w=2.1, phase=math.pi / 2), dict(off=(-150, 150), L=112, w=-2.1, phase=math.pi / 2)],
                                        col=(210, 210, 220)), scale="maior", t_full=30.0),
    "coracao_2bastoes": dict(shape="heart", mech="rods", palette="doce", hook="THE HEART FILLS UP UNTIL IT EXPLODES", hook_pt="O coração enche até explodir", bg=(60, 20, 50), RB=8.5,
                             rods=dict(rods=[dict(off=(-160, -45), L=108, w=2.3, phase=0.4), dict(off=(160, -45), L=108, w=-2.3, phase=-0.4)],
                                       col=(225, 205, 255)), scale="maior", t_full=30.0),
    "circulo_4bastoes": dict(shape="circle", mech="rods", palette="neon", hook="WAIT FOR THE ENDING!", hook_pt="Espera até o final!", bg=(50, 20, 40),
                             rods=dict(rods=[dict(off=(0, -150), L=100, w=2.4, phase=0.0), dict(off=(150, 0), L=100, w=-2.4, phase=0.0),
                                             dict(off=(0, 150), L=100, w=2.4, phase=0.0), dict(off=(-150, 0), L=100, w=-2.4, phase=0.0)],
                                       col=(120, 230, 160)), scale="menor", t_full=30.0),
    "hexagono_3bastoes": dict(shape="hexagon", mech="rods", palette="vivo", hook="3 BARS SPINNING NONSTOP", hook_pt="3 bastões girando sem parar", bg=(30, 30, 60),
                              rods=dict(rods=[dict(off=(140 * math.cos(a), 140 * math.sin(a)), L=104, w=2.2 * (1 if i % 2 == 0 else -1), phase=a)
                                              for i, a in enumerate([-math.pi / 2, math.pi / 6, 5 * math.pi / 6])],
                                        col=(230, 200, 120)), scale="maior", t_full=30.0),
    "triangulo_engrenagens": dict(shape="triangle", mech="gears", palette="neon", hook="GEARS MULTIPLYING BALLS", hook_pt="Engrenagens multiplicando bolinhas", bg=(20, 30, 55), cy=600,
                                  gears=dict(axis_deg=0, Rc=50.0, colA=(60, 160, 255), colB=(230, 200, 120), omega=1.5, teeth=12),
                                  scale="menor", RB=8.0, fill=0.88, t_full=31.0),
    "retangulo_2bastoes": dict(shape="rounded", mech="rods", palette="doce", hook="2 GIANT SPINNING BARS", hook_pt="2 bastões gigantes girando", bg=(20, 40, 30),
                               rods=dict(rods=[dict(off=(0, -165), L=150, w=1.9, phase=0.0), dict(off=(0, 165), L=150, w=-1.9, phase=0.0)],
                                         col=(200, 220, 240), th=18.0), scale="menor", t_full=30.0),
    "engrenagens_circulo": dict(shape="circle", mech="gears", palette="vivo", hook="WATCH IT MULTIPLY!",
                                gears=dict(axis_deg=0, colA=(70, 180, 230), colB=(190, 190, 200)), scale="maior"),
    "engrenagens_quadrado": dict(shape="square", mech="gears", palette="neon", hook="HOW FULL CAN IT GET?", bg=(50, 20, 40),
                                 gears=dict(axis_deg=90, colA=(200, 120, 255), colB=(80, 220, 120), omega=1.4), scale="menor"),
    "arcos_hexagono": dict(shape="hexagon", mech="arcs", palette="doce", hook="ATÉ ONDE AGUENTA?", bg=(30, 30, 60),
                           arcs=dict(RA=100.0, colA=(200, 190, 230)), scale="maior"),
    "engrenagens_coracao": dict(shape="heart", mech="gears", palette="doce", hook="FILLING THE HEART", bg=(60, 20, 50),
                                gears=dict(axis_deg=0, Rc=72.0, colA=(150, 120, 255), colB=(220, 200, 255), omega=1.3), scale="maior", RB=8.5),
    "arcos_octogono": dict(shape="octagon", mech="arcs", palette="neon", hook="THE ARCS THAT MULTIPLY", bg=(20, 40, 30),
                           arcs=dict(RA=105.0, colA=(120, 230, 160)), scale="menor"),
    "engrenagens_retangulo": dict(shape="rounded", mech="gears", palette="vivo", hook="GEARS IN A BOX", bg=(20, 30, 55),
                                  gears=dict(axis_deg=-35, colA=(60, 160, 255), colB=(230, 200, 120), omega=1.5, teeth=16), scale="menor"),
    # ---- 2026-11-10 ----
    "prensas_4": dict(shape="circle", mech="pieces", palette="vivo", theme="minimal", seeds=4, bg_min=(28, 14, 8), grid=True,
                      num_only=True, RB=6.0, maxper=5, hook="4 METAL PRESSES MULTIPLYING BALLS", hook_pt="4 prensas de metal multiplicando bolinhas",
                      pieces=dict(hub=False, col=(175, 180, 188), pieces=[
                          dict(off=(0, -262), pts=[(-65, -14), (65, -14), (65, 14), (-65, 14)], fill=True, th=8, press=dict(dir=(0, 1), amp=150, period=1.5, ph=0.0)),
                          dict(off=(0, 262), pts=[(-65, -14), (65, -14), (65, 14), (-65, 14)], fill=True, th=8, press=dict(dir=(0, -1), amp=150, period=1.5, ph=0.0)),
                          dict(off=(-262, 0), pts=[(-14, -65), (14, -65), (14, 65), (-14, 65)], fill=True, th=8, press=dict(dir=(1, 0), amp=150, period=1.5, ph=0.5)),
                          dict(off=(262, 0), pts=[(-14, -65), (14, -65), (14, 65), (-14, 65)], fill=True, th=8, press=dict(dir=(-1, 0), amp=150, period=1.5, ph=0.5))]),
                      scale="menor", t_full=30.0),
    "rotor_cunhas": dict(shape="octagon", mech="pieces", palette="manchas", theme="minimal", inherit=True, seeds=4, RB=7.5,
                         hook="2 SPINNING BLADES MULTIPLYING BALLS", hook_pt="2 pás girando multiplicam as bolinhas",
                         pieces=dict(col=(180, 184, 190), pieces=[
                             dict(pts=[(28 * math.cos(-0.45), 28 * math.sin(-0.45))] + [(250 * math.cos(a), 250 * math.sin(a)) for a in np.linspace(-0.3, 0.3, 6)] +
                                  [(28 * math.cos(0.45), 28 * math.sin(0.45))], fill=True, th=6, w=1.45, phase=ph) for ph in (0.0, math.pi)]),
                         scale="maior", t_full=30.0),
    "moldura_girando": dict(shape="circle", mech="pieces", palette="manchas", theme="minimal", inherit=True, seeds=4, RB=8.0, fill=0.9,
                            hook="THE SPINNING FRAME THAT MULTIPLIES", hook_pt="A moldura que gira e multiplica",
                            pieces=dict(col=(195, 198, 205), pieces=[
                                dict(pts=[(-70, -125), (-205, -125), (-205, 125), (-70, 125)], closed=False, th=14, w=0.85),
                                dict(pts=[(70, -125), (205, -125), (205, 125), (70, 125)], closed=False, th=14, w=0.85, phase=0.0),
                                dict(pts=[(-205, 0), (205, 0)], closed=False, th=12, w=0.85)]),
                            scale="menor", t_full=30.0),
    "estrela_quadrado": dict(shape="star", mech="pieces", palette="manchas", theme="minimal", inherit=True, seeds=5, RB=7.0, seed_r=120,
                             hook="A SPINNING SQUARE INSIDE A STAR", hook_pt="Um quadrado girando dentro da estrela",
                             pieces=dict(hub=False, col=(200, 202, 208), pieces=[
                                 dict(pts=[(-80, -80), (80, -80), (80, 80), (-80, 80)], closed=True, th=14, w=1.7)]),
                             scale="maior", t_full=30.0),
    "triangulo_quica": dict(shape="triangle", mech="bounce", palette="neon", hook="1 BALL BOUNCING IN A TRIANGLE", hook_pt="1 bolinha quicando no triângulo",
                            bg=(55, 25, 45), RB=8.0, scale="menor", t_full=37.0, cool=1.5, maxper=1),

}


def score(st, frames_n):
    if st["final"] is None or st["t_over"] is None:
        return -1e9
    return -abs(st["t_over"] - 34) - max(0, st["t_expl"] - 45) * 3


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    cfg = CONFIGS[name]
    od = os.path.join(HERE, f"out_{name}"); os.makedirs(od, exist_ok=True)
    if mode == "search":
        for seed in range(int(sys.argv[3]), int(sys.argv[4])):
            _, au, pops, st, ar, mech, CAP = simulate(cfg, seed, keep=False)
            print(name, "seed", seed, "CAP", CAP, "cheio", st["t_over"] and round(st["t_over"], 1), "explode", st["t_expl"] and round(st["t_expl"], 1), "final", st["final"], flush=True)
        sys.exit()
    seed = int(sys.argv[3])
    frames, audio, pops, st, ar, mech, CAP = simulate(cfg, seed)
    print(name, "duracao", len(frames) / FPS, "CAP", CAP, "cheio", st["t_over"], "explode", st["t_expl"], "final", st["final"], flush=True)
    rd = Renderer(cfg, ar, mech, cfg.get("RB", 9.0)); srng = np.random.default_rng(1)
    if mode == "preview":
        want = set(int(x) for x in sys.argv[4:])
        for i, fr in enumerate(frames):
            if i in want:
                cv2.imwrite(f"{od}/p_{i:04d}.png", rd.render(fr, pops, st, srng, CAP))
    else:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", f"{od}/v_tmp.mp4"], stdin=subprocess.PIPE)
        for fr in frames:
            enc.stdin.write(rd.render(fr, pops, st, srng, CAP).tobytes())
        enc.stdin.close(); enc.wait()
        dur = len(frames) / FPS
        make_audio(sorted(audio), dur, f"{od}/audio.wav", cfg.get("scale", "maior"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{od}/v_tmp.mp4", "-i", f"{od}/audio.wav", "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(dur), f"{od}/tmp.mp4"], check=True)
        os.replace(f"{od}/tmp.mp4", f"{od}/{name}.mp4")
        print("ok", f"{od}/{name}.mp4", flush=True)
