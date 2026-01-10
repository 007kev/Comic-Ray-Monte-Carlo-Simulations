#!/usr/bin/env python3
"""
cosmic_mc_predefined.py

Monte Carlo acceptance + predicted coincidence rate vs angle
for a two-scintillator cosmic ray telescope.

All parameters are PREDEFINED to match the lab setup.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PREDEFINED PARAMETERS
# -----------------------------
WIDTH_CM     = 5.0     # scintillator width (cm)
LENGTH_CM    = 10.0    # scintillator length (cm)
SEPARATION_CM = 35.0   # separation between paddles (cm)
N_EVENTS     = 100000  # MC events for acceptance
NBINS        = 19      # angles: -90 to +90 in 10° steps
MINUTES      = 30.0    # minutes per angle
A            = 1.0     # cos^2 amplitude
B            = 0.0     # background
SEED         = 1234

FLUX_NORM = 1.04       # lab normalization constant (matches partner code)

# -----------------------------
# Monte Carlo acceptance at q=0
# -----------------------------
rng = np.random.default_rng(SEED)

# sample cos(theta) for dN/dΩ ∝ cos^2(theta)
u = rng.random(N_EVENTS)
cost = u ** (1.0 / 3.0)
sint = np.sqrt(1.0 - cost**2)
phi = 2.0 * math.pi * rng.random(N_EVENTS)

tant = sint / cost
tantx = tant * np.sin(phi)
tanty = tant * np.cos(phi)

# random entry on top scintillator
xtop = rng.random(N_EVENTS) * WIDTH_CM
ytop = rng.random(N_EVENTS) * LENGTH_CM

# project to bottom scintillator
xbot = xtop - SEPARATION_CM * tantx
ybot = ytop - SEPARATION_CM * tanty

hits = (xbot >= 0) & (xbot <= WIDTH_CM) & (ybot >= 0) & (ybot <= LENGTH_CM)
N_hit = np.count_nonzero(hits)

acceptance = N_hit / N_EVENTS
d_acceptance = math.sqrt(N_hit) / N_EVENTS

area = WIDTH_CM * LENGTH_CM
R0 = FLUX_NORM * area * acceptance
dR0 = FLUX_NORM * area * d_acceptance

# -----------------------------
# Predict rate vs angle
# -----------------------------
angles = np.linspace(-90, 90, NBINS)
angles_rad = np.radians(angles)

rate = (A * np.cos(angles_rad)**2 + B) * R0
rate_err = (A * np.cos(angles_rad)**2) * dR0

# -----------------------------
# Print results
# -----------------------------
print("MC acceptance at q = 0:")
print(f"  acceptance = {acceptance:.6f} ± {d_acceptance:.6f}")
print(f"  base rate R0 = {R0:.4f} ± {dR0:.4f} counts/min\n")

print("Angle (deg)   Rate (counts/min)")
for a, r in zip(angles, rate):
    print(f"{a:6.1f}        {r:8.4f}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.errorbar(angles, rate, yerr=rate_err, fmt='o', capsize=3)
plt.xlabel("Tilt angle q (degrees)")
plt.ylabel("Coincidence rate (counts/min)")
plt.title("Monte Carlo prediction: cosmic-ray rate vs angle")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
