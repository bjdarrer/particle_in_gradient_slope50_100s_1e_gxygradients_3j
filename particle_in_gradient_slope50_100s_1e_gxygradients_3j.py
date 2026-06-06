#!/usr/bin/env python3
"""
By Brendan Darrer and ChatGPT 5.5 - Date: 6 June 2026 19:03 BST

Python conversion of Mathematica notebook:
    particle_in_gradient_slope50_100s_1a.nb
Updated as Python version 1e with explicit MP4 export options
and separate G-, X-, and Y-gradient forcing terms in dG, dX, and dY.

Model G soliton particle moving under G-gradient, X-gradient, and Y-gradient forcing.

This script implements the active 1D PDE section of the notebook in the
potential variables pGb, pXb, pYb.  The earlier Mathematica symbolic
nondimensionalisation is preserved here as comments/docstrings; the numerical
system is solved by finite differences + scipy.integrate.solve_ivp.

Mathematica residual form used in the notebook:

    -d_t pG + lap pG - v grad pG - q pG + r pX + Astep(x,t) = 0
    -d_t pX + g lap pX - v grad pX + p pG - (1+b)pX + u pY
       - s[(pX+X0)^3 - X0^3] + [(pX+X0)^2(pY+Y0)-X0^2Y0] = 0
    -d_t pY + h lap pY - v grad pY + b pX - u pY
       + s[(pX+X0)^3 - X0^3] - [(pX+X0)^2(pY+Y0)-X0^2Y0] = 0

Hence, after moving terms to the RHS:

    d_t pG = lap pG - v grad pG - q pG + r pX + Astep(x,t)
    d_t pX = g lap pX - v grad pX + p pG - (1+b)pX + u pY
             - s[(pX+X0)^3 - X0^3] + [(pX+X0)^2(pY+Y0)-X0^2Y0]
    d_t pY = h lap pY - v grad pY + b pX - u pY
             + s[(pX+X0)^3 - X0^3] - [(pX+X0)^2(pY+Y0)-X0^2Y0]

The Mathematica notebook settings were:
    G gradient = -1/50
    X gradient = -1/50
    Y gradient = -1/50
    L = 20
    T = 500
    duration = 100 seconds, fps = 1
    a=14, b=29, g=1, h=12, p=1, q=1, r=0.1, s=0, u=0, v=0, w=0
    initial potentials all zero
    periodic boundary condition p(-L/2,t)=p(L/2,t)

Outputs:
    - PNG snapshots
    - optional MP4 animation if imageio is available
    - CSV diagnostics of centre-of-mass-like estimates

Note for Brendan only:
    python3 -m venv tf-env
    source tf-env/bin/activate
    
==============

Brendan: Can you produce the file again, this time to also include a gradient of -1/50 for the Y gradient in dY, alongside the X and G gradients, and include it so that the gradients can be switched or varied it later?

ChatGPT:

I’ll update the last X-gradient version so the three fields have separate g_gradient, x_gradient, and y_gradient parameters, each defaulting to -1/50 and controllable from the command line.

Thought for 34s

Done — I produced the new version based on your attached 1d script.

Run with all three gradients at -1/50:

python3 particle_in_gradient_slope50_100s_1e_gxygradients_3j.py --outdir out_particle_gxy_3j

Vary/switch them like this:

python3 particle_in_gradient_slope50_100s_1e_gxygradients_3j.py --g-gradient -0.02 --x-gradient 0 --y-gradient -0.02

python3 particle_in_gradient_slope50_100s_1e_gxygradients_3j.py --outdir out_particle_gxy_3j --g-gradient 0 --x-gradient 0 --y-gradient 0.038

==============

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


@dataclass
class Params:
    # Model G nondimensional parameters, matching Mathematica `subs`.
    a: float = 14.0
    b: float = 29.0
    g: float = 1.0
    h: float = 12.0
    p: float = 1.0
    q: float = 1.0
    r: float = 0.1
    s: float = 0.0
    u: float = 0.0
    w: float = 0.0

    # Gradient forcing and domain.
    # Original G-gradient forcing used in dG.
    #gradient: float = -1.0 / 50.0
    gradient: float = 0.0
    #gradient: float = -1.0 / 50.0
    
    # X-gradient forcing used in dX.
    # Set to 0.0 to switch it off.
    #x_gradient: float = -1.0 / 30.0
    #x_gradient: float = -1.0 / 500.0
    x_gradient: float = 0.0

    # Y-gradient forcing used in dY.
    # Set to 0.0 to switch it off.
    #y_gradient: float = -1.0 / 50.0
    #y_gradient: float = 0.0
    y_gradient: float = 1.0 / 20  # 1/20
    y_gradient: float = 0.038
    
    L: float = 20.0
    T: float = 500.0

    # Numerical settings.
    nx: int = 401
    n_frames: int = 101
    method: str = "BDF"
    rtol: float = 1e-5
    atol: float = 1e-7


def bell(s: float, x: np.ndarray | float) -> np.ndarray | float:
    """Mathematica bell[s_,x_] := Exp[-(x/s)^2/2]."""
    return np.exp(-0.5 * (np.asarray(x) / s) ** 2)


def homogeneous_state(par: Params) -> tuple[float, float, float]:
    """
    Homogeneous steady state values from the Mathematica notebook:

        Gb0 = (a + r w)/(q - r p)
        Xb0 = (p a + q w)/(q - r p)
        Yb0 = (p a + q w) ( s (p a + q w)^2 + b(q-rp)^2 ) /
              ( (q-rp) ( (p a + q w)^2 + u(q-rp)^2 ) )

    These are the background concentrations about which pG, pX, pY are potentials.
    """
    denom = par.q - par.r * par.p
    if abs(denom) < 1e-14:
        raise ValueError("q - r*p is too close to zero; homogeneous state is singular.")
    Gb0 = (par.a + par.r * par.w) / denom
    Xb0 = (par.p * par.a + par.q * par.w) / denom
    Yb0 = (
        (par.p * par.a + par.q * par.w)
        * (par.s * (par.p * par.a + par.q * par.w) ** 2 + par.b * denom**2)
        / (denom * ((par.p * par.a + par.q * par.w) ** 2 + par.u * denom**2))
    )
    return Gb0, Xb0, Yb0


def astep(x: np.ndarray, t: float, par: Params) -> np.ndarray:
    """
    Mathematica:
        Astep[x_,t_] := -bell[1,x] bell[10,t-10] + gradient x

    This is added to the pG residual, so it appears as forcing in d_t pG.
    """
    return -bell(1.0, x) * bell(10.0, t - 10.0) + par.gradient * x


def xstep(x: np.ndarray, t: float, par: Params) -> np.ndarray:
    """
    New X-gradient forcing term.

    This mirrors the linear-gradient part of astep(), but is added to dX
    rather than dG.  It is deliberately kept separate from the original
    Gaussian seed used in astep(), so the X equation receives only the
    imposed background slope by default:

        Xstep[x_, t_] := x_gradient x

    With x_gradient = -1/50 this gives a downward X-gradient analogous to
    the original G-gradient.
    """
    return par.x_gradient * x


def ystep(x: np.ndarray, t: float, par: Params) -> np.ndarray:
    """
    New Y-gradient forcing term.

    This mirrors the linear-gradient part of astep(), but is added to dY
    rather than dG or dX:

        Ystep[x_, t_] := y_gradient x

    With y_gradient = -1/50 this gives a downward Y-gradient analogous to
    the G- and X-gradient terms.
    """
    return par.y_gradient * x


def periodic_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order periodic finite-difference Laplacian in 1D."""
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2


def periodic_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    """Second-order periodic central derivative in 1D. Kept for nonzero v experiments."""
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def rhs(t: float, y: np.ndarray, x: np.ndarray, dx: float, par: Params) -> np.ndarray:
    """Right-hand side for solve_ivp."""
    n = x.size
    pG = y[0:n]
    pX = y[n : 2 * n]
    pY = y[2 * n : 3 * n]

    Gb0, Xb0, Yb0 = homogeneous_state(par)

    lapG = periodic_laplacian(pG, dx)
    lapX = periodic_laplacian(pX, dx)
    lapY = periodic_laplacian(pY, dx)

    # Mathematica used v -> {0,0,0}; in 1D we allow scalar v=0 by default.
    v = 0.0
    gradG = periodic_gradient(pG, dx)
    gradX = periodic_gradient(pX, dx)
    gradY = periodic_gradient(pY, dx)

    cubic_shift = (pX + Xb0) ** 3 - Xb0**3
    xy_shift = (pX + Xb0) ** 2 * (pY + Yb0) - Xb0**2 * Yb0

    dG = lapG - v * gradG - par.q * pG + par.r * pX + astep(x, t, par)
    dX = (
        par.g * lapX
        - v * gradX
        + par.p * pG
        - (1.0 + par.b) * pX
        + par.u * pY
        - par.s * cubic_shift
        + xy_shift
        + xstep(x, t, par)
    )
    dY = (
        par.h * lapY
        - v * gradY
        + par.b * pX
        - par.u * pY
        + par.s * cubic_shift
        - xy_shift
        + ystep(x, t, par)
    )

    return np.concatenate([dG, dX, dY])


def make_frame(x: np.ndarray, pG: np.ndarray, pX: np.ndarray, pY: np.ndarray, t: float, out: Path) -> None:
    """Matplotlib equivalent of the Mathematica Plot frame[t_]."""
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.plot(x, pG, label=r"$\phi_G$", linewidth=2)
    ax.plot(x, pX / 10.0, label=r"$\phi_X/10$", linewidth=2)
    ax.plot(x, pY, label=r"$\phi_Y$", linewidth=2)
    ax.set_xlabel("Space")
    ax.set_ylabel("G, X, Y concentration potentials")
    ax.set_title(f"Particle with G- (slope 0), X- (slope 0), and Y- (slope 0.038) gradients, t={t:.2f}")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def diagnostics(x: np.ndarray, sol_y: np.ndarray, t_eval: np.ndarray, out_csv: Path) -> None:
    """
    Save simple tracking diagnostics.  These are not in the Mathematica notebook,
    but are useful for checking whether the localized disturbance drifts under the imposed gradients.
    """
    n = x.size
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "max_abs_pG_x", "max_abs_pX_x", "max_abs_pY_x", "pG_min", "pG_max", "pX_min", "pX_max", "pY_min", "pY_max"])
        for k, t in enumerate(t_eval):
            pG = sol_y[0:n, k]
            pX = sol_y[n : 2 * n, k]
            pY = sol_y[2 * n : 3 * n, k]
            writer.writerow([
                float(t),
                float(x[np.argmax(np.abs(pG))]),
                float(x[np.argmax(np.abs(pX))]),
                float(x[np.argmax(np.abs(pY))]),
                float(pG.min()), float(pG.max()),
                float(pX.min()), float(pX.max()),
                float(pY.min()), float(pY.max()),
            ])


def run(par: Params, outdir: Path, make_mp4: bool = True, fps: int = 10, keep_frames: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Endpoint=False gives a clean periodic grid: x[-1] is one dx before +L/2.
    x = np.linspace(-par.L / 2.0, par.L / 2.0, par.nx, endpoint=False)
    dx = x[1] - x[0]
    y0 = np.zeros(3 * par.nx)
    t_eval = np.linspace(0.0, par.T, par.n_frames)

    print("Homogeneous background Gb0, Xb0, Yb0 =", homogeneous_state(par))
    print(f"G-gradient forcing in dG = {par.gradient:g}")
    print(f"X-gradient forcing in dX = {par.x_gradient:g}")
    print(f"Y-gradient forcing in dY = {par.y_gradient:g}")
    print(f"Solving {3*par.nx} ODEs on nx={par.nx}, dx={dx:.5g}, T={par.T} using {par.method}...")

    sol = solve_ivp(
        lambda t, y: rhs(t, y, x, dx, par),
        (0.0, par.T),
        y0,
        method=par.method,
        t_eval=t_eval,
        rtol=par.rtol,
        atol=par.atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    n = par.nx
    frame_paths: list[Path] = []
    for k, t in enumerate(sol.t):
        pG = sol.y[0:n, k]
        pX = sol.y[n : 2 * n, k]
        pY = sol.y[2 * n : 3 * n, k]
        frame_path = outdir / f"frame_{k:04d}.png"
        make_frame(x, pG, pX, pY, float(t), frame_path)
        frame_paths.append(frame_path)

    diagnostics(x, sol.y, sol.t, outdir / "diagnostics.csv")

    if make_mp4:
        mp4_path = outdir / "particle_in_gradient_slope50_100s_1e_gxygradients_3j.mp4"
        try:
            import imageio.v2 as imageio

            # macro_block_size=16 avoids odd-size H.264 compatibility warnings.
            # codec/libx264 gives a normal MP4 playable in VLC, browsers, etc.
            with imageio.get_writer(
                mp4_path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=16,
            ) as writer:
                for fp in frame_paths:
                    writer.append_data(imageio.imread(fp))
            print("Wrote MP4 video:", mp4_path)
        except Exception as exc:  # pragma: no cover - optional dependency path
            print("MP4 creation failed:", exc)
            print("Try installing video support with one of:")
            print("  python3 -m pip install imageio imageio-ffmpeg")
            print("  sudo apt install ffmpeg python3-imageio")

    if not keep_frames:
        for fp in frame_paths:
            fp.unlink(missing_ok=True)
        print("Deleted individual PNG frames after writing video.")

    print("Wrote output to", outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model G particle with separate G-, X-, and Y-gradient forcing")
    parser.add_argument("--outdir", default="out_particle_in_gradient", help="Output directory")
    parser.add_argument("--nx", type=int, default=401, help="Number of periodic grid points")
    parser.add_argument("--T", type=float, default=500.0, help="Final nondimensional time")
    parser.add_argument("--frames", type=int, default=101, help="Number of output frames")
    parser.add_argument("--method", default="BDF", choices=["BDF", "Radau", "RK23", "RK45", "DOP853", "LSODA"], help="solve_ivp method")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for MP4 export")
    parser.add_argument("--g-gradient", type=float, default=-1.0/50.0, help="Linear G-gradient forcing added to dG")
    parser.add_argument("--x-gradient", type=float, default=-1.0/50.0, help="Linear X-gradient forcing added to dX")
    parser.add_argument("--y-gradient", type=float, default=-1.0/50.0, help="Linear Y-gradient forcing added to dY")
    parser.add_argument("--no-mp4", action="store_true", help="Do not attempt MP4 animation export")
    parser.add_argument("--delete-frames", action="store_true", help="Delete PNG frames after MP4 creation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = Params(nx=args.nx, T=args.T, n_frames=args.frames, method=args.method, rtol=args.rtol, atol=args.atol, gradient=args.g_gradient, x_gradient=args.x_gradient, y_gradient=args.y_gradient)
    run(params, Path(args.outdir), make_mp4=not args.no_mp4, fps=args.fps, keep_frames=not args.delete_frames)
