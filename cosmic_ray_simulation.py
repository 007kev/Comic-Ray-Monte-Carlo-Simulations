#!/usr/bin/env python3
"""
Cosmic-ray telescope (two scintillators) Monte Carlo with cos^2(theta) flux.
Defaults set to the user's lab setup:
  - nbins = 9 (angles from -pi/2 .. +pi/2)
  - minutes = 30 per angle
  - scintillator size = 5 cm x 10 cm
  - separation d = 35 cm

Run:
  python crt_cosmic_mc.py --plot
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

def sample_direction_cos3(n):
    """
    Sample n directions on the upper hemisphere with PDF ~ cos^3(theta).
    Return unit vectors (vx, vy, vz) in LAB frame (z vertical).
    """
    u = np.random.random(size=n)
    c = np.power(1.0 - u, 0.25)              # cos(theta)
    s = np.sqrt(np.maximum(0.0, 1.0 - c*c))  # sin(theta)
    phi = 2.0 * math.pi * np.random.random(size=n)
    vx = s * np.cos(phi)
    vy = s * np.sin(phi)
    vz = c
    return np.stack([vx, vy, vz], axis=1)

def rot_x(vecs, angle):
    """Rotate vectors by 'angle' around x-axis."""
    ca, sa = math.cos(angle), math.sin(angle)
    x = vecs[:,0]
    y = ca*vecs[:,1] - sa*vecs[:,2]
    z = sa*vecs[:,1] + ca*vecs[:,2]
    return np.stack([x,y,z], axis=1)

def simulate_at_angle(q, n_throws, size_x, size_y, separation_d):
    """
    Tilt telescope by +q around x-axis. Geometry in telescope frame:
      top slab at z'=0, bottom at z'=+d.
    1) Sample LAB directions with cos^3 theta.
    2) Rotate into telescope frame by -q.
    3) Uniform (x',y') over top slab; propagate to z'=d and test inside bottom slab.
    """
    V_lab = sample_direction_cos3(n_throws)
    V_tel = rot_x(V_lab, -q)
    mask = V_tel[:,2] > 0.0
    V_tel = V_tel[mask]
    n = V_tel.shape[0]
    if n == 0:
        return 0

    x0 = (np.random.random(size=n) - 0.5) * size_x
    y0 = (np.random.random(size=n) - 0.5) * size_y

    dx = separation_d * (V_tel[:,0] / V_tel[:,2])
    dy = separation_d * (V_tel[:,1] / V_tel[:,2])
    x2 = x0 + dx
    y2 = y0 + dy

    hits = np.count_nonzero((np.abs(x2) <= size_x/2) & (np.abs(y2) <= size_y/2))
    return int(hits)

def fit_A_from_vertical(count_vertical, minutes, B):
    """From vertical bin (q≈0): model rate0 = A + B => A = rate0 - B."""
    rate0 = count_vertical / minutes
    return max(0.0, rate0 - B)

def main():
    p = argparse.ArgumentParser(description="Monte Carlo: 2-slab CRT with cos^2(theta) flux")
    # --- Defaults updated to your setup ---
    p.add_argument("--nbins", type=int, default=9, help="Number of tilt bins [-pi/2, +pi/2]")
    p.add_argument("--minutes", type=float, default=30.0, help="Exposure per angle (minutes)")
    p.add_argument("--throws-per-min", type=int, default=200_000,
                   help="MC throws per minute of exposure (controls precision)")
    p.add_argument("--size-x", type=float, default=5.0, help="Scintillator size in x (cm)")
    p.add_argument("--size-y", type=float, default=10.0, help="Scintillator size in y (cm)")
    p.add_argument("--separation", type=float, default=35.0, help="Separation d (cm)")
    p.add_argument("--B", type=float, default=0.0, help="Constant background (counts/min)")
    p.add_argument("--A", type=float, default=None,
                   help="Signal amplitude for model A*cos^2(q)+B; if omitted, fit from q≈0 bin.")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed")
    p.add_argument("--plot", action="store_true", help="Show a plot with error bars")
    args = p.parse_args()

    rng = np.random.default_rng()

    # Angle bins (centers)
    q_edges = np.linspace(-0.5*math.pi, 0.5*math.pi, args.nbins+1)
    q_centers = 0.5*(q_edges[:-1] + q_edges[1:])
    i0 = int(np.argmin(np.abs(q_centers)))  # index of bin closest to 0 (vertical)

    minutes = args.minutes
    throws_per_angle = int(args.throws_per_min * minutes)

    # Geometry-only true coincidences
    true_counts = []
    for q in q_centers:
        hits = simulate_at_angle(
            q,
            n_throws=throws_per_angle,
            size_x=args.size_x,
            size_y=args.size_y,
            separation_d=args.separation
        )
        true_counts.append(hits)
    true_counts = np.array(true_counts, dtype=float)

    # Impose cos^2(q) shape on signal + add constant B background
    cos2 = np.cos(q_centers)**2
    cos2_ref = max(1e-12, cos2[i0])

    if true_counts[i0] > 0:
        scaled_signal = true_counts * (cos2 / cos2_ref)
    else:
        total_geom = np.sum(true_counts)
        scaled_signal = (total_geom * cos2 / np.sum(cos2)) if total_geom > 0 else np.zeros_like(true_counts)

    B = max(0.0, float(args.B))
    bg_counts = rng.poisson(lam=B * minutes, size=args.nbins).astype(float)

    observed = np.clip(np.rint(scaled_signal) + bg_counts, 0, None).astype(int)
    obs_err = np.sqrt(np.maximum(1.0, observed))
    plot_histogram_angles(q_centers, observed, minutes)

    rate = observed / minutes
    rate_err = obs_err / minutes

    if args.A is None:
        A = fit_A_from_vertical(observed[i0], minutes, B)
    else:
        A = max(0.0, float(args.A))
    model_rate = A * cos2 + B

    print("# q(rad)    q(deg)    counts   rate_per_min   err_per_min")
    for qi, qd, c, r, dr in zip(q_centers, np.degrees(q_centers), observed, rate, rate_err):
        print(f"{qi:+.5f}  {qd:+7.2f}   {int(c):6d}    {r:12.6f}   {dr:11.6f}")

    print("\n# Model (overlay): rate(q) = A*cos^2(q) + B")
    print(f"# A = {A:.6f} counts/min,  B = {B:.6f} counts/min")
    print("# Geometry: size_x = %.1f cm, size_y = %.1f cm, separation d = %.1f cm" %
          (args.size_x, args.size_y, args.separation))
    print("# Exposure per angle: %.2f min, Throws/angle: %d" % (minutes, throws_per_angle))

    if args.plot:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.errorbar(np.degrees(q_centers), rate, yerr=rate_err, fmt='o', label="Observed (MC)")
        ax.plot(np.degrees(q_centers), model_rate, lw=2, label="Model A cos^2(q) + B")
        ax.set_xlabel("Telescope tilt q (degrees, q=0 vertical)")
        ax.set_ylabel("Coincidence rate (counts/min)")
        ax.set_title("Cosmic-ray telescope: MC vs. A cos²(q) + B")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
def plot_histogram_angles(q_centers, observed, minutes):
    """
    Make a histogram-style plot of observed rates per tilt angle.
    """
    import matplotlib.pyplot as plt
    rate = observed / minutes
    
    plt.figure(figsize=(8,5))
    plt.bar(np.degrees(q_centers), rate, width=10, align='center', edgecolor='black', alpha=0.7)
    plt.xlabel("Tilt angle q (degrees)")
    plt.ylabel("Rate (counts/min)")
    plt.title("Monte Carlo cosmic telescope: histogram of rates per angle")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
