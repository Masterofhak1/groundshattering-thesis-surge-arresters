"""
Visual Oustaloup RC Network Synthesis with Foster Topologies
============================================================

Implements Oustaloup method with Foster I and Foster II network realizations
for surge arrester characterisation in high voltage engineering.

Context: High Voltage Electrical Engineering - Surge Arrester Characterisation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Literal


@dataclass
class ColeColeParams:
    """Cole-Cole model parameters for surge arrester characterisation"""

    f0: float  # Lower frequency bound [Hz]
    f1: float  # Upper frequency bound [Hz]
    N: int  # Number of RC branches
    alpha: float  # Cole-Cole exponent (0 < α < 1)
    tau0: float  # Characteristic time [s]
    C_inf: float  # High-frequency capacitance [F]
    C0: float  # Low-frequency capacitance [F]


@dataclass
class FosterNetwork:
    """RC network component values for Foster topologies"""

    R: np.ndarray  # Resistance values [Ω]
    C: np.ndarray  # Capacitance values [F]
    C_series: float  # Series capacitance (C0 - C∞) [F]
    C_parallel: float  # Parallel capacitance (C∞) [F]
    G_parallel: float  # Parallel conductance (Foster II) [S]
    R_series: float = 0.0  # Series resistance R∞ (Foster I) [Ω]
    topology: str = ""  # 'Foster_I' or 'Foster_II'
    poles: np.ndarray = None  # Pole frequencies [Hz]
    zeros: np.ndarray = None  # Zero frequencies [Hz]


def calculate_crossing_frequencies(params: ColeColeParams) -> np.ndarray:
    """Calculate crossing frequencies where piecewise approximation intersects ideal impedance."""
    i = np.arange(2 * params.N + 1)
    f_ci = params.f0 * (params.f1 / params.f0) ** (i / (2 * params.N))
    return f_ci


def crossing_to_poles_zeros(
    params: ColeColeParams, f_ci: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert crossing frequencies to pole and zero frequencies."""
    k = np.arange(1, params.N + 1)
    even_crossings = f_ci[2 * k]
    adjustment = (params.f1 / params.f0) ** (params.alpha / (2 * params.N))

    fp = even_crossings / adjustment
    fz = even_crossings * adjustment

    return fp, fz


def calculate_rc_values(params: ColeColeParams, fp: np.ndarray, fz: np.ndarray) -> dict:
    """Calculate RC component values using partial fraction expansion."""
    delta_C = params.C0 - params.C_inf
    omega_p = 2 * np.pi * fp
    omega_z = 2 * np.pi * fz

    # Calculate R0 at geometric center frequency
    f_center = np.sqrt(params.f0 * params.f1)
    omega_c = 2 * np.pi * f_center
    Z_exact_mag = (params.tau0 / delta_C) * (omega_c * params.tau0) ** (-params.alpha)

    # R0 scaling
    prod_num = np.prod(np.abs(1 + 1j * omega_c / omega_z))
    prod_den = np.prod(np.abs(1 + 1j * omega_c / omega_p))
    R0 = Z_exact_mag * (prod_den / prod_num)

    # Calculate residues (R values)
    R_values = np.zeros(params.N)
    for k in range(params.N):
        num = np.prod(1 - omega_p[k] / omega_z)
        den_terms = [1 - omega_p[k] / omega_p[m] for m in range(params.N) if m != k]
        den = np.prod(den_terms) if den_terms else 1.0
        R_values[k] = R0 * num / den

    # Calculate C values
    C_values = 1 / (omega_p * R_values)

    return {
        "R_values": R_values,
        "C_values": C_values,
        "R0": R0,
        "R_inf": R0 * np.prod(omega_p / omega_z),
    }


def synthesize_foster_network(
    params: ColeColeParams,
    fp: np.ndarray,
    fz: np.ndarray,
    rc_values: dict,
    topology: Literal["Foster_I", "Foster_II"] = "Foster_I",
) -> FosterNetwork:
    """
    Synthesize Foster I or Foster II network from Oustaloup method results.

    Foster I: Series chain of parallel R||C branches
              C∞ || [R∞ + Σ(R||C)] || Cs

    Foster II: Parallel connection of series R+C branches
               C∞ || [1/(G∞ + Σ(sC/(1+sRC)))] || Cs
    """
    delta_C = params.C0 - params.C_inf
    omega_p = 2 * np.pi * fp
    omega_z = 2 * np.pi * fz

    if topology == "Foster_I":
        # Foster I uses the same R and C values from Oustaloup synthesis
        R_values = rc_values["R_values"]
        C_values = rc_values["C_values"]
        R_series = rc_values["R_inf"]
        G_parallel = 0.0

    else:  # Foster II
        # For Foster II, we need to calculate admittance form
        R0 = rc_values["R0"]
        G_inf = (1 / R0) * np.prod(omega_p / omega_z)

        # Calculate admittance residues
        residues = np.zeros(params.N)
        for k in range(params.N):
            num = 1.0
            for m in range(params.N):
                num *= 1 - omega_z[k] / omega_p[m]

            den = 1.0
            for m in range(params.N):
                if m != k:
                    den *= 1 - omega_z[k] / omega_z[m]

            residues[k] = -(1 / R0) * num / (den * omega_z[k])

        # Foster II: C_values are the residues, R_values from time constants
        C_values = residues
        R_values = 1 / (omega_z * C_values)
        G_parallel = G_inf
        R_series = 0.0

    return FosterNetwork(
        R=R_values,
        C=C_values,
        C_series=delta_C,
        C_parallel=params.C_inf,
        G_parallel=G_parallel,
        R_series=R_series,
        topology=topology,
        poles=fp,
        zeros=fz,
    )


def construct_zigzag_approximation(
    params: ColeColeParams,
    f_ci: np.ndarray,
    fp: np.ndarray,
    fz: np.ndarray,
    rc_values: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Construct the piecewise linear (zigzag) approximation."""
    delta_C = params.C0 - params.C_inf

    # Get the ideal impedance at crossing points
    omega_ci = 2 * np.pi * f_ci
    Z_ci = (params.tau0 / delta_C) * (omega_ci * params.tau0) ** (-params.alpha)

    # Build complete frequency list
    all_points = []

    for i, (f, Z) in enumerate(zip(f_ci, Z_ci)):
        all_points.append({"f": f, "Z": Z, "type": "crossing", "idx": i})

    for i, f in enumerate(fp):
        all_points.append({"f": f, "Z": None, "type": "pole", "idx": i})

    for i, f in enumerate(fz):
        all_points.append({"f": f, "Z": None, "type": "zero", "idx": i})

    # Sort by frequency
    all_points.sort(key=lambda x: x["f"])

    # Initialize segments storage
    f_segments = []
    Z_segments = []
    segment_types = []

    # Find positions of crossing points in sorted list
    crossing_positions = []
    for i, pt in enumerate(all_points):
        if pt["type"] == "crossing":
            crossing_positions.append(i)

    # Process each interval between consecutive crossing points
    for i in range(len(crossing_positions) - 1):
        start_pos = crossing_positions[i]
        end_pos = crossing_positions[i + 1]

        fc_start = all_points[start_pos]
        fc_end = all_points[end_pos]

        # Find what's between these crossing points
        between = all_points[start_pos + 1 : end_pos]

        if not between:
            # No transitions - single segment
            log_slope = (np.log10(fc_end["Z"]) - np.log10(fc_start["Z"])) / (
                np.log10(fc_end["f"]) - np.log10(fc_start["f"])
            )

            if abs(log_slope + 1) < abs(log_slope):  # Closer to -1 than to 0
                # Capacitive segment
                f_segments.append([fc_start["f"], fc_end["f"]])
                Z_segments.append([fc_start["Z"], fc_end["Z"]])
                segment_types.append("C")
            else:
                # Resistive segment
                Z_mean = np.sqrt(fc_start["Z"] * fc_end["Z"])
                f_segments.append([fc_start["f"], fc_end["f"]])
                Z_segments.append([Z_mean, Z_mean])
                segment_types.append("R")

        else:
            # We have transitions between crossings
            first_trans = between[0]

            if first_trans["type"] == "pole":
                # Pattern: Capacitive → Pole → Resistive
                pole_f = first_trans["f"]

                # 1. Capacitive from fc_start to pole
                Z_at_pole = fc_start["Z"] * (fc_start["f"] / pole_f)
                f_segments.append([fc_start["f"], pole_f])
                Z_segments.append([fc_start["Z"], Z_at_pole])
                segment_types.append("C")

                # 2. Resistive from pole to fc_end
                f_segments.append([pole_f, fc_end["f"]])
                Z_segments.append([fc_end["Z"], fc_end["Z"]])
                segment_types.append("R")

            elif first_trans["type"] == "zero":
                # Pattern: Resistive → Zero → Capacitive
                zero_f = first_trans["f"]

                # 1. Resistive from fc_start to zero
                f_segments.append([fc_start["f"], zero_f])
                Z_segments.append([fc_start["Z"], fc_start["Z"]])
                segment_types.append("R")

                # 2. Capacitive from zero to fc_end
                Z_at_zero = fc_end["Z"] * (fc_end["f"] / zero_f)
                f_segments.append([zero_f, fc_end["f"]])
                Z_segments.append([Z_at_zero, fc_end["Z"]])
                segment_types.append("C")

    return f_segments, Z_segments, segment_types


def plot_oustaloup_approximation(
    params: ColeColeParams,
    f_ci: np.ndarray,
    fp: np.ndarray,
    fz: np.ndarray,
    rc_values: dict,
    save_path: str = "oustaloup_approximation.png",
    show_vertical_lines: bool = False,  # NEW PARAMETER
    show_fci_labels: bool = False,      # NEW PARAMETER
):
    """
    Plot the Oustaloup approximation with clean professional style.
    
    Parameters:
    -----------
    show_vertical_lines : bool
        If True, adds vertical dashed lines at crossing frequencies and poles/zeros
    show_fci_labels : bool
        If True, adds fc0, fc1, fc2... labels at top of vertical lines
    """
    # Frequency array for plotting
    f = np.logspace(np.log10(params.f0 / 10), np.log10(params.f1 * 10), 1000)
    omega = 2 * np.pi * f

    # Exact fractional impedance ZA
    delta_C = params.C0 - params.C_inf
    Z_exact = (params.tau0 / delta_C) * (omega * params.tau0) ** (-params.alpha)

    # Calculate actual RC network approximation
    Z_approx = np.zeros_like(omega, dtype=complex)
    for R, C in zip(rc_values["R_values"], rc_values["C_values"]):
        Z_approx += R / (1 + 1j * omega * R * C)

    # Calculate zigzag segments
    f_segments, Z_segments, segment_types = construct_zigzag_approximation(
        params, f_ci, fp, fz, rc_values
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    fig.subplots_adjust(hspace=0.3)

    # Color scheme
    color_ideal = "#2E7D32"  # Dark green
    color_approx = "#1565C0"  # Dark blue
    color_crossing = "#C62828"  # Dark red
    color_poles = "#1976D2"  # Blue
    color_zeros = "#F57C00"  # Orange

    # ============= Upper plot - Overview =============
    
    # NEW: Add vertical lines if requested
    if show_vertical_lines:
        # Get y-axis range for label positioning
        y_min = min(np.abs(Z_exact)) * 0.5
        y_max = max(np.abs(Z_exact)) * 2
        
        # Vertical lines at crossing frequencies
        for i, fc in enumerate(f_ci):
            ax1.axvline(x=fc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            if show_fci_labels:
                ax1.text(fc, y_max*0.8, f'$f_{{c{i}}}$', 
                        ha='center', va='bottom', fontsize=10)
        
        # Vertical lines at poles
        for i, fp_val in enumerate(fp):
            ax1.axvline(x=fp_val, color=color_poles, linestyle=':', linewidth=1.0, alpha=0.6)
            if show_fci_labels:
                ax1.text(fp_val, y_min*1.3, f'$f_{{p{i+1}}}$', 
                        ha='center', va='top', fontsize=10, color=color_poles)
        
        # Vertical lines at zeros
        for i, fz_val in enumerate(fz):
            ax1.axvline(x=fz_val, color=color_zeros, linestyle=':', linewidth=1.0, alpha=0.6)
            if show_fci_labels:
                ax1.text(fz_val, y_min*1.3, f'$f_{{z{i+1}}}$', 
                        ha='center', va='top', fontsize=10, color=color_zeros)
    
    # Plot the curves
    ax1.loglog(
        f,
        np.abs(Z_exact),
        color=color_ideal,
        linewidth=2.5,
        label=f"Ideal Cole-Cole",
        zorder=3
    )
    ax1.loglog(
        f,
        np.abs(Z_approx),
        color=color_approx,
        linestyle="--",
        linewidth=2,
        label=f"RC Network",  # IMPROVED LABEL
        zorder=3
    )

    # Mark crossing frequencies
    Z_exact_at_crossings = (params.tau0 / delta_C) * (
        2 * np.pi * f_ci * params.tau0
    ) ** (-params.alpha)
    ax1.scatter(
        f_ci,
        Z_exact_at_crossings,
        c=color_crossing,
        s=100,
        edgecolors="white",
        linewidth=1.5,
        zorder=5,
        label="Crossing Points",
    )

    # Mark poles and zeros
    Z_at_poles = (params.tau0 / delta_C) * (2 * np.pi * fp * params.tau0) ** (
        -params.alpha
    )
    Z_at_zeros = (params.tau0 / delta_C) * (2 * np.pi * fz * params.tau0) ** (
        -params.alpha
    )

    ax1.scatter(
        fp,
        Z_at_poles,
        c=color_poles,
        s=80,
        marker="^",
        edgecolors="white",
        linewidth=1.5,
        zorder=5,
        label="Poles",
    )
    ax1.scatter(
        fz,
        Z_at_zeros,
        c=color_zeros,
        s=80,
        marker="v",
        edgecolors="white",
        linewidth=1.5,
        zorder=5,
        label="Zeros",
    )

    ax1.set_xlabel("Frequency (Hz)", fontsize=10)
    ax1.set_ylabel("|Z| (Ω)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # ============= Lower plot - Zigzag detail =============
    
    # NEW: Add vertical lines if requested
    if show_vertical_lines:
        for fc in f_ci:
            ax2.axvline(x=fc, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        for fp_val in fp:
            ax2.axvline(x=fp_val, color=color_poles, linestyle=':', linewidth=0.5, alpha=0.3)
        for fz_val in fz:
            ax2.axvline(x=fz_val, color=color_zeros, linestyle=':', linewidth=0.5, alpha=0.3)
    
    ax2.loglog(
        f,
        np.abs(Z_exact),
        color=color_ideal,
        linewidth=2.5,
        alpha=0.3,
        label="Ideal Cole-Cole",
        zorder=2
    )
    ax2.loglog(
        f,
        np.abs(Z_approx),
        color=color_approx,
        linewidth=1.5,
        linestyle=":",
        alpha=0.6,
        label="RC Network",
        zorder=2
    )

    # Draw the zigzag segments
    resistive_labeled = False
    capacitive_labeled = False

    for f_seg, Z_seg, seg_type in zip(f_segments, Z_segments, segment_types):
        if seg_type == "R":
            if not resistive_labeled:
                ax2.loglog(
                    f_seg,
                    Z_seg,
                    "#FDD835",
                    linewidth=4,
                    alpha=0.8,
                    solid_capstyle="round",
                    zorder=4,
                    label="Resistive segments",
                )
                resistive_labeled = True
            else:
                ax2.loglog(
                    f_seg,
                    Z_seg,
                    "#FDD835",
                    linewidth=4,
                    alpha=0.8,
                    solid_capstyle="round",
                    zorder=4,
                )
        else:  # Capacitive
            if not capacitive_labeled:
                ax2.loglog(
                    f_seg,
                    Z_seg,
                    "#E53935",
                    linewidth=4,
                    alpha=0.8,
                    solid_capstyle="round",
                    zorder=4,
                    label="Capacitive segments",
                )
                capacitive_labeled = True
            else:
                ax2.loglog(
                    f_seg,
                    Z_seg,
                    "#E53935",
                    linewidth=4,
                    alpha=0.8,
                    solid_capstyle="round",
                    zorder=4,
                )

    # Mark crossing points
    ax2.scatter(
        f_ci,
        Z_exact_at_crossings,
        c=color_crossing,
        s=120,
        edgecolors="white",
        linewidth=2,
        zorder=5,
        label="Crossing points",
    )

    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.set_ylabel("|Z| (Ω)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_foster_topologies(
    params: ColeColeParams,
    foster_I: FosterNetwork,
    foster_II: FosterNetwork,
    save_path: str = "foster_topologies.png",
):
    """Plot Foster I and Foster II network frequency responses."""
    # Frequency array
    f = np.logspace(np.log10(params.f0 / 10), np.log10(params.f1 * 10), 1000)
    omega = 2 * np.pi * f

    # Exact fractional element ZA
    delta_C = params.C0 - params.C_inf
    Z_A_exact = (params.tau0 / delta_C) * (1j * omega * params.tau0) ** (-params.alpha)

    # Foster I approximation
    Z_A_foster_I = foster_I.R_series * np.ones_like(omega, dtype=complex)
    for R, C in zip(foster_I.R, foster_I.C):
        Z_A_foster_I += R / (1 + 1j * omega * R * C)

    # Foster II approximation
    Y_A_foster_II = foster_II.G_parallel * np.ones_like(omega, dtype=complex)
    for R, C in zip(foster_II.R, foster_II.C):
        Y_A_foster_II += 1j * omega * C / (1 + 1j * omega * R * C)
    Z_A_foster_II = 1 / Y_A_foster_II

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    fig.subplots_adjust(hspace=0.3)

    # Magnitude plot
    ax1.loglog(f, np.abs(Z_A_exact), "k-", linewidth=2.5, label="Exact ZA")
    ax1.loglog(f, np.abs(Z_A_foster_I), "b--", linewidth=2, label="Foster I")
    ax1.loglog(f, np.abs(Z_A_foster_II), "r:", linewidth=2, label="Foster II")

    # Mark poles and zeros
    for fp in foster_I.poles:
        ax1.axvline(fp, color="blue", linestyle=":", alpha=0.2, linewidth=0.8)
    for fz in foster_I.zeros:
        ax1.axvline(fz, color="red", linestyle=":", alpha=0.2, linewidth=0.8)

    ax1.set_xlabel("Frequency (Hz)", fontsize=10)
    ax1.set_ylabel("|ZA| (Ω)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Phase plot
    ax2.semilogx(
        f, np.angle(Z_A_exact) * 180 / np.pi, "k-", linewidth=2.5, label="Exact ZA"
    )
    ax2.semilogx(
        f, np.angle(Z_A_foster_I) * 180 / np.pi, "b--", linewidth=2, label="Foster I"
    )
    ax2.semilogx(
        f, np.angle(Z_A_foster_II) * 180 / np.pi, "r:", linewidth=2, label="Foster II"
    )
    ax2.axhline(
        y=-params.alpha * 90,
        color="gray",
        linestyle=":",
        label=f"Theory = {-params.alpha*90:.1f}°",
    )

    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.set_ylabel("Phase (degrees)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim([params.f0 / 10, params.f1 * 10])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_complete_models(
    params: ColeColeParams,
    foster_I: FosterNetwork,
    foster_II: FosterNetwork,
    save_path: str = "complete_capacitor_models.png",
):
    """Plot complete capacitor models including C∞ and Cs."""
    # Frequency array
    f = np.logspace(np.log10(params.f0 / 10), np.log10(params.f1 * 10), 1000)
    omega = 2 * np.pi * f

    # Exact complete Cole-Cole impedance
    delta_C = params.C0 - params.C_inf
    Z_exact = (params.tau0 / delta_C) * (1j * omega * params.tau0) ** (-params.alpha)
    Y_exact_total = 1j * omega * params.C_inf + 1 / Z_exact
    Z_exact_with_Cinf = 1 / Y_exact_total
    Z_exact_complete = Z_exact_with_Cinf + 1 / (1j * omega * delta_C)

    # Foster I complete model
    Z_foster_I = foster_I.R_series * np.ones_like(omega, dtype=complex)
    for R, C in zip(foster_I.R, foster_I.C):
        Z_foster_I += R / (1 + 1j * omega * R * C)
    Y_with_Cinf = 1 / Z_foster_I + 1j * omega * foster_I.C_parallel
    Z_foster_I_complete = 1 / Y_with_Cinf + 1 / (1j * omega * foster_I.C_series)

    # Foster II complete model
    Y_foster_II = foster_II.G_parallel * np.ones_like(omega, dtype=complex)
    for R, C in zip(foster_II.R, foster_II.C):
        Y_foster_II += 1j * omega * C / (1 + 1j * omega * R * C)
    Z_A_approx = 1 / Y_foster_II
    Y_with_Cinf = 1 / Z_A_approx + 1j * omega * foster_II.C_parallel
    Z_foster_II_complete = 1 / Y_with_Cinf + 1 / (1j * omega * foster_II.C_series)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot all three
    ax.loglog(f, np.abs(Z_exact_complete), "k-", linewidth=2.5, label="Exact Cole-Cole")
    ax.loglog(
        f,
        np.abs(Z_foster_I_complete),
        "b--",
        linewidth=2,
        label=f"Foster I (N={params.N})",
    )
    ax.loglog(
        f,
        np.abs(Z_foster_II_complete),
        "r:",
        linewidth=2,
        label=f"Foster II (N={params.N})",
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("|Z| (Ω)", fontsize=10)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_synthesis_results(
    params: ColeColeParams, foster_I: FosterNetwork, foster_II: FosterNetwork
):
    """Print synthesis results for thesis documentation."""
    print("\n" + "=" * 70)
    print("OUSTALOUP RC NETWORK SYNTHESIS")
    print("Surge Arrester Frequency-Dependent Characterisation")
    print("=" * 70)

    print(f"\nCole-Cole Model Parameters:")
    print(f"  Frequency range: {params.f0:.1f} Hz to {params.f1:.1f} Hz")
    print(f"  Number of RC branches: {params.N}")
    print(f"  Fractional exponent α: {params.alpha:.3f}")
    print(f"  Characteristic time τ₀: {params.tau0*1e3:.1f} ms")
    print(f"  C₀ = {params.C0*1e9:.3f} nF")
    print(f"  C∞ = {params.C_inf*1e9:.3f} nF")
    print(f"  ΔC = {(params.C0-params.C_inf)*1e9:.3f} nF")

    print("\n" + "-" * 50)
    print("FOSTER I TOPOLOGY")
    print("Circuit: C∞ || [R∞ + Σ(Rk||Ck)] in series with Cs")
    print("-" * 50)
    print(f"R∞ = {foster_I.R_series/1e6:.3f} MΩ")
    print("\nRC Branches:")
    for k, (R, C) in enumerate(zip(foster_I.R, foster_I.C)):
        print(f"  Branch {k+1}: R{k+1} = {R/1e6:8.3f} MΩ || C{k+1} = {C*1e12:8.1f} pF")
        print(
            f"           τ{k+1} = {R*C*1e3:8.3f} ms, fp{k+1} = {foster_I.poles[k]:8.1f} Hz"
        )

    print("\n" + "-" * 50)
    print("FOSTER II TOPOLOGY")
    print("Circuit: C∞ || [1/(G∞ + Σ(sCk/(1+sRkCk)))] in series with Cs")
    print("-" * 50)
    print(
        f"G∞ = {foster_II.G_parallel*1e9:.3f} nS = 1/{1/foster_II.G_parallel/1e6:.1f} MΩ"
    )
    print("\nRC Branches:")
    for k, (R, C) in enumerate(zip(foster_II.R, foster_II.C)):
        print(
            f"  Branch {k+1}: C{k+1} = {C*1e6:8.5f} μF in series with R{k+1} = {R/1e6:8.2f} MΩ"
        )
        print(
            f"           τ{k+1} = {R*C*1e3:8.3f} ms, fz{k+1} = {foster_II.zeros[k]:8.1f} Hz"
        )

    print("\n" + "-" * 50)
    print("EXTERNAL COMPONENTS (both topologies):")
    print(f"  Parallel capacitor C∞ = {params.C_inf*1e6:.4f} μF")
    print(f"  Series capacitor Cs = {(params.C0-params.C_inf)*1e6:.4f} μF")


# Example usage
if __name__ == "__main__":
    # Define Cole-Cole parameters from surge arrester measurements
    params = ColeColeParams(
        f0=1,  # Hz - Lower measurement frequency
        f1=1000.0,  # Hz - Upper measurement frequency
        N=3,  # Number of RC branches
        alpha=0.587,  # Cole-Cole exponent
        tau0=0.1,  # s - Characteristic relaxation time
        C_inf=0.521e-9,  # F - High-frequency capacitance
        C0=0.882e-9,  # F - Low-frequency capacitance
    )

    # Oustaloup synthesis
    f_ci = calculate_crossing_frequencies(params)
    fp, fz = crossing_to_poles_zeros(params, f_ci)
    rc_values = calculate_rc_values(params, fp, fz)

    # Create Foster networks
    foster_I = synthesize_foster_network(params, fp, fz, rc_values, topology="Foster_I")
    foster_II = synthesize_foster_network(
        params, fp, fz, rc_values, topology="Foster_II"
    )

    # Print synthesis results
    print_synthesis_results(params, foster_I, foster_II)

    # Generate and save plots
    print("\nGenerating plots...")
    plot_oustaloup_approximation(
    params, f_ci, fp, fz, rc_values,
    save_path="oustaloup_with_lines_and_labels.png",
    show_vertical_lines=True,
    show_fci_labels=True
)
    print("  Saved: oustaloup_zigzag.png")

    plot_foster_topologies(
        params, foster_I, foster_II, save_path="foster_topologies_comparison.png"
    )
    print("  Saved: foster_topologies_comparison.png")

    plot_complete_models(
        params, foster_I, foster_II, save_path="complete_capacitor_models.png"
    )
    print("  Saved: complete_capacitor_models.png")

    print("\nAll plots saved successfully!")

