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
