# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Ethan Earl - https://github.com/ec175/spectrometry_public
"""stl_tunnel.py - put a 3-D model (STL/OBJ/PLY/GLB) in the 2-D wind tunnel.

The solver is 2-D, so the model has to become ONE closed profile first. Two ways to cut it:

    side view  (default)  the whole model's silhouette seen from the side: fuselage, tail, rider,
                          everything. The 2-D flow is the flow round that silhouette EXTRUDED
                          infinitely deep, so treat the numbers as a shape comparison, not as
                          the 3-D aircraft's drag.
    --slice C             a planar cross-section at depth coordinate C (e.g. through a wing,
                          which gives you that wing's aerofoil section). This is the textbook
                          2-D wing-section experiment.

Either cut is rendered to a PNG cut-out and handed to `shapes.image_body`, so it goes through
the engine's own tracing, sub-cell gap welding and texture mapping: the body the solver sees and
the body drawn on screen are the same polygon by construction.

    python stl_tunnel.py model.stl --still 2 4 6         # tuning stills
    python stl_tunnel.py model.stl --preview             # 960x540 / 30 fps
    python stl_tunnel.py model.stl                       # 1920x1080 / 60 fps
    python stl_tunnel.py model.stl --slice 6 --aoa 6     # wing section at 6 deg

Axes: `--forward` is the model axis the nose points along, `--up` is the model's up. The flow
comes from the left and meets the nose first.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wt import shapes                                                       # noqa: E402
from wt.config import OUT, RenderConfig, encoder_args, find_ffmpeg          # noqa: E402
from wt.gpu import GPU                                                      # noqa: E402
from wt.render import Tunnel                                                # noqa: E402


class PaddedConfig(RenderConfig):
    """RenderConfig with an off-screen margin UPSTREAM and DOWNSTREAM only.

    `overscan` pads all four sides by the same fraction. A zoomed-in, centred body needs room
    in the streamwise direction only, where the inlet clamp and the outlet sponge would
    otherwise sit a few cells from it. A tall frame with the flow running across it already
    has plenty of cross-stream room, so padding that too would double the cost for nothing.
    `pad_up` / `pad_down` are fractions of the visible streamwise extent.
    """
    pad_up = 0.0
    pad_down = 0.0

    @property
    def nx(self) -> int:
        return self.vis_nx + int(round(self.vis_nx * (self.pad_up + self.pad_down)))

    @property
    def ny(self) -> int:
        return self.vis_ny

    @property
    def vis_x0(self) -> int:
        return int(round(self.vis_nx * self.pad_up))

    @property
    def vis_y0(self) -> int:
        return 0

AX = {"x": 0, "y": 1, "z": 2}


def _axis(s):
    s = s.strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    return AX[s.lstrip("+-")], sign


# --- DXF rib -> smooth aerofoil -----------------------------------------------------------
def _cst_basis(x, order):
    """Kulfan CST: class function sqrt(x)(1-x) times Bernstein polynomials, plus a linear term
    for trailing-edge thickness / chord-line tilt. Round nose and a clean TE by construction."""
    from math import comb
    c = np.sqrt(x) * (1.0 - x)
    cols = [c * comb(order, i) * x ** i * (1.0 - x) ** (order - i) for i in range(order + 1)]
    return np.stack(cols + [x], 1)


def dxf_rib_profile(path, rib=0, order=8, margin=0.012):
    """A laser-cut rib (outline with spar slots, LE/TE notches, plus the insert pieces) ->
    the smooth aerofoil it was cut from, as a closed polygon in unit chord (LE at 0, +y up).

    The notches are where the spars and edge stock were INSERTED, so they are not part of the
    section the air sees. The rib outline is sampled column by column, every column within
    `margin` chord of an insert is dropped, and a CST curve is least-squares fitted to what is
    left, separately for the upper and lower surfaces.
    """
    import ezdxf
    from ezdxf import path as dpath
    doc = ezdxf.readfile(path)
    pieces = []
    for e in doc.modelspace():
        subs = e.virtual_entities() if e.dxftype() == "INSERT" else [e]
        for v in subs:
            try:
                p = np.array([(q.x, q.y) for q in dpath.make_path(v).flattening(0.05)])
            except Exception:
                continue
            if len(p) > 2:
                pieces.append(p)
    box = lambda p: (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())
    area = lambda p: np.ptp(p[:, 0]) * np.ptp(p[:, 1])
    big = max(area(p) for p in pieces)
    ribs = sorted([p for p in pieces if area(p) > 0.25 * big], key=lambda p: p[:, 0].min())
    if not ribs:
        raise SystemExit("no rib outline found in the DXF")
    R = ribs[int(rib)]
    rx0, rx1, ry0, ry1 = box(R)
    tol = 0.005 * (rx1 - rx0)          # an insert belongs to this rib if its CENTRE is on it
    ctr = lambda p: (0.5 * (box(p)[0] + box(p)[1]), 0.5 * (box(p)[2] + box(p)[3]))
    ins = [p for p in pieces if area(p) <= 0.25 * big and
           rx0 - tol < ctr(p)[0] < rx1 + tol and ry0 - tol < ctr(p)[1] < ry1 + tol]
    allx = np.concatenate([R[:, 0]] + [p[:, 0] for p in ins])
    le_x, te_x = allx.min(), allx.max()
    chord = te_x - le_x
    le_piece = min(ins, key=lambda p: p[:, 0].min()) if ins else None
    if le_piece is not None and le_piece[:, 0].min() <= rx0:
        le_y = 0.5 * (le_piece[:, 1].min() + le_piece[:, 1].max())
    else:
        le_y = None

    # column envelope of the rib outline, at ~0.5 mm (well under 0.1% chord)
    s = 2.0
    W, H = int(np.ceil((rx1 - rx0) * s)) + 3, int(np.ceil((ry1 - ry0) * s)) + 3
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon([((x - rx0) * s + 1, (ry1 - y) * s + 1) for x, y in R], fill=255)
    m = np.asarray(img) > 0
    cols = np.nonzero(m.any(0))[0]
    xs = (cols - 1) / s + rx0
    top = ry1 - (np.argmax(m[:, cols], 0) - 1) / s
    bot = ry1 - (H - 1 - np.argmax(m[::-1, cols], 0) - 1) / s
    keep = np.ones(len(xs), bool)
    for p in ins:
        keep &= ~((xs > p[:, 0].min() - margin * chord) & (xs < p[:, 0].max() + margin * chord))
    keep &= (xs > le_x + margin * chord) & (xs < te_x - margin * chord)
    if le_y is None:
        le_y = 0.5 * (top[keep][0] + bot[keep][0])
    xc = (xs[keep] - le_x) / chord
    A = _cst_basis(xc, order)
    cu = np.linalg.lstsq(A, (top[keep] - le_y) / chord, rcond=None)[0]
    cl = np.linalg.lstsq(A, (bot[keep] - le_y) / chord, rcond=None)[0]
    res = max(np.abs(A @ cu - (top[keep] - le_y) / chord).max(),
              np.abs(A @ cl - (bot[keep] - le_y) / chord).max())

    b = np.linspace(0.0, np.pi, 240)
    xe = 0.5 * (1.0 - np.cos(b))                       # cosine spacing: dense at LE and TE
    B = _cst_basis(xe, order)
    yu, yl = B @ cu, B @ cl
    poly = np.concatenate([np.stack([xe[::-1], yu[::-1]], 1), np.stack([xe[1:], yl[1:]], 1)])
    t = yu - yl
    info = dict(mode=f"rib {int(rib) + 1}/{len(ribs)} smoothed (CST order {order})",
                ribs=len(ribs), inserts=len(ins), chord_units=chord,
                thickness=float(t.max()), thickness_at=float(xe[np.argmax(t)]),
                te_thickness=float(t[-1]), fit_max_err=float(res),
                raw=((R - [le_x, le_y]) / chord))
    return poly, info


def profile_cutout(poly, out_png, px=2400, color=(214, 206, 188)):
    """Unit-chord polygon (+y up) -> RGBA cut-out, nose on the left, for `image_body`."""
    pad = 8
    k = px - 1
    y1 = poly[:, 1].max()
    W, H = px + 2 * pad, int(np.ceil((y1 - poly[:, 1].min()) * k)) + 2 * pad
    alpha = Image.new("L", (W, H), 0)
    ImageDraw.Draw(alpha).polygon([(x * k + pad, (y1 - y) * k + pad) for x, y in poly], fill=255)
    im = Image.new("RGB", (W, H), color).convert("RGBA")
    im.putalpha(alpha)
    im.save(out_png)


# --- model -> cut-out PNG ---------------------------------------------------------------
def model_cutout(path, out_png, forward="+y", up="+z", slice_at=None, px=2400, rib=0):
    """Render the model's side silhouette (or a cross-section) as an RGBA cut-out.

    Image x runs nose -> tail (left to right), image y runs top -> bottom. Returns a dict of
    what was cut, for the log. A .dxf is read as a sheet of laser-cut ribs instead.
    """
    if path.lower().endswith(".dxf"):
        poly, info = dxf_rib_profile(path, rib=rib)
        profile_cutout(poly, out_png, px)
        info.update(length=1.0, height=float(np.ptp(poly[:, 1])), png=out_png, poly=poly)
        return info
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    fi, fs = _axis(forward)
    ui, us = _axis(up)
    di = ({0, 1, 2} - {fi, ui}).pop()                     # the depth axis we look along
    V = np.asarray(mesh.vertices, np.float64)
    u = -fs * V[:, fi]                                    # nose on the LEFT (smallest u)
    v = -us * V[:, ui]                                    # up is image-UP (smaller row)

    if slice_at is None:
        u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
    else:
        normal = np.zeros(3); normal[di] = 1.0
        origin = np.zeros(3); origin[di] = float(slice_at)
        segs = trimesh.intersections.mesh_plane(mesh, normal, origin)   # (n, 2, 3)
        if len(segs) == 0:
            lo, hi = V[:, di].min(), V[:, di].max()
            raise SystemExit(f"--slice {slice_at} misses the model: its depth axis "
                             f"({'xyz'[di]}) spans {lo:.3f} .. {hi:.3f}")
        su, sv = -fs * segs[:, :, fi], -us * segs[:, :, ui]
        u0, u1, v0, v1 = su.min(), su.max(), sv.min(), sv.max()

    span = max(u1 - u0, 1e-9)
    k = (px - 1) / span                                   # image px per model unit
    pad = 8
    W, H = px + 2 * pad, int(np.ceil((v1 - v0) * k)) + 2 * pad

    def to_img(uu, vv):
        return (uu - u0) * k + pad, (vv - v0) * k + pad

    if slice_at is None:
        # painter's algorithm, far faces first, lit from the viewer and above
        T = V[mesh.faces]
        n = np.asarray(mesh.face_normals, np.float64)
        depth = T[:, :, di].mean(1)
        order = np.argsort(depth)                         # viewer sits on the +depth side
        light = np.zeros(3); light[di] = 0.75; light[ui] = 0.55 * us; light[fi] = -0.35 * fs
        light /= np.linalg.norm(light)
        lam = np.abs(n @ light)
        shade = np.clip(70 + 170 * lam, 0, 255)
        tu, tv = to_img(-fs * T[:, :, fi], -us * T[:, :, ui])
        rgb = Image.new("RGB", (W, H), (0, 0, 0))
        alpha = Image.new("L", (W, H), 0)
        dr, da = ImageDraw.Draw(rgb), ImageDraw.Draw(alpha)
        for f in order:
            poly = list(zip(tu[f].tolist(), tv[f].tolist()))
            g = int(shade[f])
            dr.polygon(poly, fill=(g, g + 4 if g < 251 else 255, min(255, g + 10)))
            da.polygon(poly, fill=255)
        info = dict(mode="side silhouette", faces=len(T))
    else:
        from scipy.ndimage import binary_closing, binary_fill_holes
        line = Image.new("L", (W, H), 0)
        dl = ImageDraw.Draw(line)
        a, b = to_img(su, sv)
        for i in range(len(segs)):
            dl.line([(a[i, 0], b[i, 0]), (a[i, 1], b[i, 1])], fill=255, width=3)
        m = np.asarray(line) > 0
        m = binary_fill_holes(binary_closing(m, iterations=6))   # welds internal ribs/slits
        alpha = Image.fromarray((m * 255).astype(np.uint8))
        rgb = Image.new("RGB", (W, H), (205, 210, 216))
        info = dict(mode=f"section at {'xyz'[di]}={slice_at:g}", segments=len(segs))

    im = rgb.convert("RGBA")
    im.putalpha(alpha)
    im.save(out_png)
    info.update(length=span, height=(v1 - v0), png=out_png)
    return info


# --- the composition ----------------------------------------------------------------------
class ModelInTunnel:
    """One rigid body held in the stream: at a fixed incidence, or swept slowly through a range
    (the rig turning the model - a legitimate input, see the engine README)."""

    def __init__(self, body, chord, cx, cy, aoa, sweep, duration, pivot, title):
        self.pts, self.rgb, self.ybox = body["points"], body["rgb"], body["ybox"]
        self.chord, self.cx, self.cy = float(chord), float(cx), float(cy)
        self.aoa0, self.sweep = float(aoa), sweep
        self.duration, self.pivot, self.title = float(duration), float(pivot), title
        self._aoa = self.aoa_at(0.0)

    def ref_len(self):
        return self.chord

    def aoa_at(self, t):
        if self.sweep is None:
            return self.aoa0
        # legs of equal length between the listed angles, each eased so the model comes to rest
        # at every turning point: a step in angular rate would be a step in wall speed
        legs = len(self.sweep) - 1
        s = np.clip(t / max(self.duration, 1e-9), 0.0, 1.0) * legs
        k = min(int(s), legs - 1)
        a0, a1 = self.sweep[k], self.sweep[k + 1]
        return a0 + (a1 - a0) * (0.5 - 0.5 * np.cos(np.pi * (s - k)))

    def bodies(self, t):
        self._aoa = self.aoa_at(t)
        return [shapes.place(self.pts, self.chord, self.cx, self.cy, self._aoa, self.pivot)]

    def body_textures(self):
        return [(self.rgb, shapes.place_uv(self.chord, self.cx, self.cy, self._aoa,
                                           ylim=self.ybox, pivot=self.pivot))]


def wall_force(lbm):
    """Momentum-exchange force on every solid in the lattice -> (Fx, Fy), lattice units.

    Same link sum as `LBM.force()`, with one difference that matters: the population heading
    INTO the wall is taken post-collision, which at the end of a step is the value that has just
    streamed into the solid cell, rather than the pre-collision value still at the fluid node.
    Checked against an independent control-volume momentum budget (sum of c_x^2 f across an
    upstream and a downstream plane): cylinder Re 20 Cd 2.427 vs 2.418, Re 100 1.335 vs 1.330,
    NACA 0012 Re 1000 0.136 vs 0.136. `LBM.force()` read 3.80, 3.40 and 1.48 on the same runs.
    """
    from wt.gpu import asnumpy, xp
    from wt.lbm import CX, CY, OPP
    f, solid = lbm.f, lbm.solid
    fx = fy = 0.0
    for k in range(1, 9):
        sh = (-int(CY[k]), -int(CX[k]))
        link = (~solid) & xp.roll(solid, sh, axis=(0, 1))
        amt = float(asnumpy(((xp.roll(f[k], sh, axis=(0, 1)) + f[int(OPP[k])]) * link).sum()))
        fx += float(CX[k]) * amt
        fy += float(CY[k]) * amt
    return fx, fy


def _font(px):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              r"C:\Windows\Fonts\consola.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(p, px)
        except Exception:
            continue
    return ImageFont.load_default()


def hud(img, lines, font):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    x, y = int(im.width * 0.025), int(im.height * 0.035)
    for s in lines:
        d.text((x + 2, y + 2), s, font=font, fill=(0, 0, 0))
        d.text((x, y), s, font=font, fill=(236, 240, 236))
        y += int(font.size * 1.4)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--name", default=None, help="output stem (default: model file name)")
    ap.add_argument("--forward", default="+y", help="model axis the nose points along")
    ap.add_argument("--up", default="+z", help="model up axis")
    ap.add_argument("--title", default=None, help="text shown on screen (default: file name)")
    ap.add_argument("--rib", type=int, default=0, help="DXF only: which rib, 0 = leftmost")
    ap.add_argument("--slice", type=float, default=None,
                    help="cut a cross-section at this depth coordinate instead of the side view")
    ap.add_argument("--aoa", type=float, default=0.0, help="angle of attack, deg (nose up +)")
    ap.add_argument("--sweep", default=None, help="sweep incidence a0:a1[:a2...] over the clip, deg (equal legs)")
    ap.add_argument("--pivot", type=float, default=0.35, help="rotation point, fraction of length")
    ap.add_argument("--chord", type=float, default=0.34, help="body length / visible width")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--field", default="speed", choices=["speed", "vort", "pressure", "stag"])
    ap.add_argument("--u0", type=float, default=0.08)
    ap.add_argument("--re", type=float, default=4000.0)
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--preview", action="store_true", help="960x540 at 30 fps, same lattice")
    ap.add_argument("--vertical", action="store_true", help="1080x1920 (9:16)")
    ap.add_argument("--flow", default=None, choices=["up", "down", "right"],
                    help="flow direction on screen (default: up when --vertical, else right)")
    ap.add_argument("--center", action="store_true",
                    help="centre the body in frame and simulate off-screen margins up/downstream")
    ap.add_argument("--pad", type=float, default=0.6,
                    help="with --center: off-screen tunnel each side, fraction of visible length")
    ap.add_argument("--film", action="store_true", help="CRT 'filmed off a screen' pass")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--still", type=float, nargs="*", default=None,
                    help="write PNG stills at these times instead of a video")
    a = ap.parse_args()

    name = a.name or os.path.splitext(os.path.basename(a.model))[0]
    if a.slice is not None and a.name is None:
        name += f"_slice{a.slice:g}"
    os.makedirs(OUT, exist_ok=True)

    W, H = (1080, 1920) if a.vertical else (1920, 1080)
    flow = a.flow or ("up" if a.vertical else "right")
    cfg = PaddedConfig(width=W, height=H, fps=60, flow=flow)
    if a.center:
        cfg.pad_up = cfg.pad_down = float(a.pad)
    if a.preview:                 # halve frame AND scale: the lattice, i.e. the physics, is unchanged
        cfg.width, cfg.height, cfg.scale, cfg.fps = cfg.width // 2, cfg.height // 2, cfg.scale / 2, 30
    cfg.u0, cfg.re, cfg.steps, cfg.field, cfg.film = a.u0, a.re, a.steps, a.field, a.film
    cfg.settle = 2.0

    cut_png = os.path.join(OUT, f"{name}_cutout.png")
    info = model_cutout(a.model, cut_png, a.forward, a.up, a.slice, rib=a.rib)
    chord = a.chord * cfg.vis_nx
    body = shapes.image_body(cut_png, cells=chord)
    ext = body["ybox"][1] - body["ybox"][0]
    print(f"model: {info['mode']}, length {info['length']:.3f}, height {info['height']:.3f} "
          f"(model units) -> {chord:.0f} x {chord * ext:.0f} lattice cells, "
          f"{len(body['points'])} outline points")

    sweep = tuple(float(s) for s in a.sweep.split(":")) if a.sweep else None
    title = a.title or (os.path.basename(a.model) +
                        (f"  |  {info['mode']}" if a.slice is not None else ""))
    if a.center:      # the body's own middle (x = 0.5 in unit space) on the frame's centre
        cx = cfg.vis_x0 + 0.5 * cfg.vis_nx - (0.5 - a.pivot) * chord
    else:
        cx = cfg.vis_x0 + 0.26 * cfg.vis_nx
    scene = ModelInTunnel(body, chord, cx=cx, cy=cfg.vis_y0 + cfg.vis_ny / 2,
                          aoa=a.aoa, sweep=sweep, duration=a.seconds, pivot=a.pivot, title=title)

    tun = Tunnel(scene, cfg)
    print(f"lattice {tun.nx}x{tun.ny}, {tun.lbm.describe()}, u0={cfg.u0} (Ma {cfg.u0 * 3 ** 0.5:.3f}),"
          f" Re={cfg.re:g} on length, {tun.steps_pf:.1f} steps/frame, "
          f"backend={'GPU' if GPU else 'CPU'}", flush=True)
    t0 = time.time()
    tun.settle(cfg.settle)
    print(f"  settled in {time.time() - t0:.0f}s", flush=True)

    font = _font(max(14, min(cfg.width, cfg.height) // 30))
    n = max(1, int(round(a.seconds * cfg.fps)))
    stills = sorted(a.still) if a.still is not None else None
    tag = "_preview" if a.preview else ""
    out_mp4 = os.path.join(OUT, f"{name}{tag}.mp4")
    tmp = out_mp4 + ".part.mp4"
    proc = None
    if stills is None:
        proc = subprocess.Popen(
            [find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0", "-an",
             *encoder_args(), "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    look = None
    if cfg.film:
        from wt.film import make_look
        look = make_look(cfg.width, cfg.height, cfg.fps, n, glow=cfg.glow, ambient=cfg.ambient,
                         bow=cfg.bow, seed=cfg.seed, exposure=cfg.exposure)

    log_path = os.path.join(OUT, f"{name}{tag}_forces.csv")
    rows, cd_s, cl_s = [], None, None
    nxt = 0
    try:
        for i in range(n + (1 if stills else 0)):
            t = i / cfg.fps
            tun.advance(t)
            fx, fy = wall_force(tun.lbm)
            q = 0.5 * cfg.u0 * cfg.u0 * chord
            cd, cl = fx / q, -fy / q                      # lattice +y is screen-DOWN: lift is -y
            k = 0.08                                      # ~0.4 s low-pass for the on-screen number
            cd_s = cd if cd_s is None else cd_s + k * (cd - cd_s)
            cl_s = cl if cl_s is None else cl_s + k * (cl - cl_s)
            rows.append((round(t, 4), round(scene._aoa, 3), cd, cl))
            want_img = proc is not None or (nxt < len(stills) and t >= stills[nxt])
            if not want_img:
                continue
            img = tun.frame(t)
            if not a.no_hud:
                img = hud(img, [title,
                                f"AoA {scene._aoa:+5.1f} deg   Re {cfg.re:,.0f}",
                                f"Cd {cd_s:6.3f}   Cl {cl_s:+6.3f}   L/D {cl_s / max(cd_s, 1e-6):+5.2f}"],
                          font)
            if look is not None:
                img = look.process(img, i)
            if proc is not None:
                proc.stdin.write(np.ascontiguousarray(img).tobytes())
            else:
                p = os.path.join(OUT, f"{name}_{t:g}s.png")
                Image.fromarray(img).save(p)
                print(f"still: {p}", flush=True)
                nxt += 1
                if nxt >= len(stills):
                    break
            if i % 10 == 0:
                h = tun.lbm.health()
                if not np.isfinite(h) or h > tun.lbm.health_limit:
                    raise RuntimeError(f"solver diverged at t={t:.2f}s (max|u|={h:.3f}). "
                                       f"Lower --u0 / --re or --chord.")
            if i % cfg.fps == 0:
                print(f"  t={t:4.1f}s  AoA {scene._aoa:+5.1f}  Cd {cd_s:.3f}  Cl {cl_s:+.3f}  "
                      f"max|u| {tun.lbm.health():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if proc is not None:
            proc.stdin.close(); proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
            os.replace(tmp, out_mp4)
            print("done:", out_mp4)
    except BaseException:
        if proc is not None:
            try:
                proc.kill(); proc.wait()
            except Exception:
                pass
            if os.path.exists(tmp):
                os.remove(tmp)
        raise
    finally:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "aoa_deg", "Cd", "Cl"])
            w.writerows(rows)
        if rows:
            tail = np.array([r[2:] for r in rows[len(rows) // 2:]])
            print(f"forces -> {log_path}  (2nd-half mean: Cd {tail[:, 0].mean():.3f}, "
                  f"Cl {tail[:, 1].mean():+.3f}; per unit depth, on length)")


if __name__ == "__main__":
    main()
