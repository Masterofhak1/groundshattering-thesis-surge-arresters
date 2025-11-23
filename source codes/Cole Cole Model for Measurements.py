"""
Cole-Cole Parameter Extraction from Validated R_eq and |X_eq| Data
In the style of Cole Cole Parameters.py but for impedance

This code:
1. Reads the validated CSV with R_eq and |X_eq| columns (in MΩ)
2. Fits Cole-Cole model to impedance data
3. Creates plots similar to Cole Cole Parameters.py
4. Saves parameters and interpolated data

IMPORTANT: Input CSV should have R_equ and |X_equ| in MΩ (megaohms)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from scipy import stats
import os


def cole_cole_impedance(freq, C_inf, C_0, tau_0, alpha):
    """
    Calculate impedance from Cole-Cole capacitance model.
    Z = 1/(jωC*) where C* = C_∞ + (C_0 - C_∞)/(1 + (jωτ)^α)
    """
    omega = 2 * np.pi * freq

    # Cole-Cole complex capacitance
    jw_tau = 1j * omega * tau_0
    C_complex = C_inf + (C_0 - C_inf) / (1 + jw_tau**alpha)

    # Impedance
    Z_complex = 1 / (1j * omega * C_complex)

    R = np.real(Z_complex)
    X = np.abs(np.imag(Z_complex))

    return R, X


def fit_cole_cole_to_validated_impedance(
    csv_file="Impedance Plots Validation.csv",
    output_dir="cole_cole_impedance_fit",
    prioritize_R=False,
):
    """
    Fit Cole-Cole model to validated impedance data.
    Following the style of Cole Cole Parameters.py

    IMPORTANT: Expects R_equ and |X_equ| columns to be in MΩ (megaohms)
    The code will convert to Ω for internal calculations

    Parameters:
    -----------
    csv_file : str
        Path to CSV file with validated impedance data
    output_dir : str
        Directory to save results
    prioritize_R : bool
        If True, prioritizes fitting R over X (useful when R fit is poor)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Read validated data
    df = pd.read_csv(csv_file)

    # Extract frequency and impedance data
    frequencies = df["Freq"].values
    # IMPORTANT: Convert from MΩ to Ω
    R_measured = df["R_equ"].values * 1e6  # MΩ to Ω
    X_measured = df["|X_equ|"].values * 1e6  # MΩ to Ω
    Z_measured = np.sqrt(R_measured**2 + X_measured**2) # Calculate |Z| from R and X


    print("Cole-Cole Parameter Extraction from Impedance Data")
    print("=" * 60)
    print(f"Number of frequency points: {len(frequencies)}")
    print(f"Frequency range: {min(frequencies)} - {max(frequencies)} Hz")

    # Display data
    print("\nMeasured Impedance Data:")
    print(f"{'Freq (Hz)':>10} {'R_eq (MΩ)':>12} {'|X_eq| (MΩ)':>12} {'|Z| (MΩ)':>12}")
    print("-" * 48)
    for i in range(len(frequencies)):
        print(
            f"{frequencies[i]:>10.0f} {R_measured[i]/1e6:>12.3f} {X_measured[i]/1e6:>12.3f} {Z_measured[i]/1e6:>12.3f}"
        )

    # Calculate slopes (for reference and plots)
    log_freq = np.log10(frequencies)
    log_R = np.log10(R_measured)
    log_X = np.log10(X_measured)
    Z_measured = np.sqrt(R_measured**2 + X_measured**2)
    log_Z = np.log10(Z_measured)

    slope_R, intercept_R, r_value_R, _, _ = stats.linregress(log_freq, log_R)
    slope_X, intercept_X, r_value_X, _, _ = stats.linregress(log_freq, log_X)
    slope_Z, _, r_value_Z, _, _ = stats.linregress(log_freq, log_Z)


    print(f"\nInitial slope analysis:")
    print(f"R slope: {slope_R:.3f} (R² = {r_value_R**2:.4f})")
    print(f"|X| slope: {slope_X:.3f} (R² = {r_value_X**2:.4f})")
    print(f"|Z| slope: {slope_Z:.3f} (R² = {r_value_Z**2:.4f})")

    # Check if R slope is very steep
    if slope_R < -1.3:
        print("\nNOTE: R has a very steep slope. Cole-Cole model may have limitations")
        print(
            "in reproducing such steep slopes. Consider alternative models if fit is poor."
        )
        print(f"Cole-Cole theoretical limits: R slope can range from -α to -(2-α)")
        print(f"Your R slope of {slope_R:.3f} suggests complex behavior.")

    # Initial parameter estimates
    # From impedance at low and high frequency
    # Z = 1/(jωC) → C ≈ 1/(ωX)
    C_low = 1 / (2 * np.pi * frequencies[0] * X_measured[0])
    C_high = 1 / (2 * np.pi * frequencies[-1] * X_measured[-1])

    # For steep R slopes, we need larger C_0/C_inf ratio
    if slope_R < -1.2:  # Steep slope
        C_0_init = C_low * 2.0  # Larger ratio
        C_inf_init = C_high * 0.5
    else:
        C_0_init = C_low * 1.2
        C_inf_init = C_high * 0.8

    # tau_0 from frequency where R ≈ X or from slope intersection
    ratio = R_measured / X_measured
    idx_unity = np.argmin(np.abs(ratio - 1))
    if idx_unity < len(frequencies) - 1:
        f_unity = frequencies[idx_unity]
    else:
        # Use frequency where slopes would intersect
        f_unity = 10 ** (-(intercept_R - intercept_X) / (slope_R - slope_X))
        f_unity = np.clip(f_unity, frequencies[0] / 10, frequencies[-1] * 10)

    tau_0_init = 1 / (2 * np.pi * f_unity)

    # Alpha affects the steepness of transitions
    # Lower alpha can help with steeper R slopes
    alpha_init = 0.6 if slope_R < -1.2 else 0.8

    print(f"\nInitial parameter estimates:")
    print(f"C_∞ = {C_inf_init*1e9:.3f} nF")
    print(f"C_0 = {C_0_init*1e9:.3f} nF")
    print(f"τ_0 = {tau_0_init*1e6:.3f} μs")
    print(f"α = {alpha_init:.3f}")

    # Define objective function with improved weighting
    def objective(params):
        C_inf, C_0, tau_0, alpha = params

        if C_0 <= C_inf:
            return 1e10

        # Calculate model impedance
        R_model, X_model = cole_cole_impedance(frequencies, C_inf, C_0, tau_0, alpha)

        # Use log-space errors for better fitting across all decades
        # This prevents large values from dominating
        R_error_log = np.mean((np.log10(R_model) - np.log10(R_measured)) ** 2)
        X_error_log = np.mean((np.log10(X_model) - np.log10(X_measured)) ** 2)

        # Also include relative errors for balance
        R_error_rel = np.mean(((R_model - R_measured) / R_measured) ** 2)
        X_error_rel = np.mean(((X_model - X_measured) / X_measured) ** 2)

        # Combined error with emphasis on getting both R and X right
        if prioritize_R:
            # Weight R much more heavily
            total_error = (
                5.0 * R_error_log + X_error_log + 2.0 * R_error_rel + 0.5 * X_error_rel
            )
        else:
            # Balanced weighting with slight R emphasis
            total_error = (
                2.0 * R_error_log + X_error_log + 0.5 * R_error_rel + 0.5 * X_error_rel
            )

        return total_error

    # Set bounds - wider to allow better exploration
    # For steep R slopes, allow lower alpha values
    alpha_min = 0.01 if slope_R < -1.3 else 0.1

    bounds = [
        (C_inf_init * 0.01, C_inf_init * 20),  # C_inf - much wider
        (C_0_init * 0.2, C_0_init * 5),  # C_0 - wider
        (1e-9, 1e-1),  # tau_0 - wider range
        (alpha_min, 0.99),  # alpha - adjusted based on R slope
    ]

    # Run optimization with multiple strategies
    print("\nRunning differential evolution optimization...")
    print("(This may take a moment for better fitting...)")

    best_result = None
    best_error = float("inf")

    # Try different strategies for better global optimization
    strategies = ["best1bin", "best2bin", "rand1bin", "currenttobest1bin"]

    for strategy in strategies:
        print(f"\nTrying strategy: {strategy}")
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            strategy=strategy,
            maxiter=5000,  # More iterations
            popsize=50,  # Larger population
            atol=1e-15,
            tol=1e-15,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,  # Polish with L-BFGS-B
            workers=1,
            disp=False,
        )

        if result.fun < best_error:
            best_error = result.fun
            best_result = result
            print(f"  Better fit found! Error: {result.fun:.6f}")

    result = best_result

    # If R fit is still poor, try a refinement stage
    if prioritize_R and result.fun > 0.1:
        print("\nRunning refinement stage for better R fitting...")

        # Use current best as starting point
        C_inf_ref, C_0_ref, tau_0_ref, alpha_ref = result.x

        # Tighter bounds around current best
        bounds_refined = [
            (C_inf_ref * 0.5, C_inf_ref * 2.0),
            (C_0_ref * 0.8, C_0_ref * 1.2),
            (tau_0_ref * 0.1, tau_0_ref * 10),
            (max(0.01, alpha_ref - 0.2), min(0.99, alpha_ref + 0.2)),
        ]

        # Refine with different algorithm settings
        result_refined = differential_evolution(
            objective,
            bounds_refined,
            seed=123,
            strategy="currenttobest1bin",
            maxiter=3000,
            popsize=100,  # Larger population for refinement
            atol=1e-18,
            tol=1e-18,
            mutation=(0.3, 1.0),  # Smaller mutations for fine-tuning
            recombination=0.95,
            polish=True,
            workers=1,
            disp=False,
        )

        if result_refined.fun < result.fun:
            result = result_refined
            print(f"Refinement improved fit! New error: {result.fun:.6f}")

    # Extract optimized parameters
    C_inf_opt, C_0_opt, tau_0_opt, alpha_opt = result.x

    print(f"\nOptimized Cole-Cole Parameters:")
    print(f"C_∞ = {C_inf_opt*1e9:.3f} nF")
    print(f"C_0 = {C_0_opt*1e9:.3f} nF")
    print(f"Δε_r = (C_0 - C_∞)/C_∞ = {(C_0_opt/C_inf_opt - 1):.3f}")
    print(f"τ_0 = {tau_0_opt*1e6:.3f} μs")
    print(f"f_0 = 1/(2πτ_0) = {1/(2*np.pi*tau_0_opt):.3f} Hz")
    print(f"α = {alpha_opt:.3f}")

    # Calculate model values
    R_model, X_model = cole_cole_impedance(
        frequencies, C_inf_opt, C_0_opt, tau_0_opt, alpha_opt
    )

    # Calculate goodness of fit
    R_squared_R = 1 - np.sum((R_measured - R_model) ** 2) / np.sum(
        (R_measured - np.mean(R_measured)) ** 2
    )
    R_squared_X = 1 - np.sum((X_measured - X_model) ** 2) / np.sum(
        (X_measured - np.mean(X_measured)) ** 2
    )

    # Also calculate R² in log space
    R_squared_R_log = 1 - np.sum(
        (np.log10(R_measured) - np.log10(R_model)) ** 2
    ) / np.sum((np.log10(R_measured) - np.mean(np.log10(R_measured))) ** 2)
    R_squared_X_log = 1 - np.sum(
        (np.log10(X_measured) - np.log10(X_model)) ** 2
    ) / np.sum((np.log10(X_measured) - np.mean(np.log10(X_measured))) ** 2)

    print(f"\nGoodness of fit:")
    print(f"R² for R: {R_squared_R:.4f} (linear), {R_squared_R_log:.4f} (log)")
    print(f"R² for |X|: {R_squared_X:.4f} (linear), {R_squared_X_log:.4f} (log)")

    # Show individual errors
    print(f"\nDetailed fit analysis:")
    print(f"{'Freq (Hz)':>10} {'R_meas':>10} {'R_model':>10} {'Error %':>10}")
    print("-" * 45)
    for i in range(len(frequencies)):
        err_pct = (R_model[i] - R_measured[i]) / R_measured[i] * 100
        print(
            f"{frequencies[i]:>10.0f} {R_measured[i]/1e6:>10.3f} {R_model[i]/1e6:>10.3f} {err_pct:>10.1f}"
        )

    # Calculate and compare slopes
    log_R_model = np.log10(R_model)
    log_X_model = np.log10(X_model)
    slope_R_model, _, _, _, _ = stats.linregress(log_freq, log_R_model)
    slope_X_model, _, _, _, _ = stats.linregress(log_freq, log_X_model)

    print(f"\nSlope comparison:")
    print(f"R slope - Measured: {slope_R:.3f}, Model: {slope_R_model:.3f}")
    print(f"X slope - Measured: {slope_X:.3f}, Model: {slope_X_model:.3f}")

    # Show theoretical slope limits for the fitted parameters
    print(f"\nTheoretical slope limits with α = {alpha_opt:.3f}:")
    print(f"R slope: -{alpha_opt:.3f} (low freq) to -{2-alpha_opt:.3f} (high freq)")
    print(f"X slope: approaches -1 (ideal capacitor)")

    if abs(slope_R_model - slope_R) > 0.2:
        print("\nWARNING: Large slope mismatch for R. The Cole-Cole model")
        print("may not be ideal for this data. Consider:")
        print("1. Using prioritize_R=True for better R fitting")
        print("2. Alternative models (e.g., distributed RC networks)")
        print("3. Checking if data represents pure dielectric behavior")

    # Create plots (in Cole Cole Parameters.py style)
    create_impedance_plots(
        frequencies,
        R_measured,
        X_measured,
        R_model,
        X_model,
        C_inf_opt,
        C_0_opt,
        tau_0_opt,
        alpha_opt,
        output_dir,
    )

    # Save results
    save_impedance_results(
        frequencies,
        R_measured,
        X_measured,
        R_model,
        X_model,
        C_inf_opt,
        C_0_opt,
        tau_0_opt,
        alpha_opt,
        output_dir,
    )

    return result.x


def create_impedance_plots(
    freq, R_measured, X_measured, R_model, X_model, C_inf, C_0, tau_0, alpha, output_dir
):
    """
    Create separate R vs f and |X| vs f plots with exact fit through measured points.
    Following the style of Cole Cole Parameters.py
    """
    # Calculate slopes and create interpolations
    log_freq = np.log10(freq)
    log_R_measured = np.log10(R_measured)
    log_X_measured = np.log10(X_measured)

    slope_R, intercept_R, r_value_R, _, _ = stats.linregress(log_freq, log_R_measured)
    slope_X, intercept_X, r_value_X, _, _ = stats.linregress(log_freq, log_X_measured)

    # Calculate |Z| = sqrt(R² + X²)
    Z_measured = np.sqrt(R_measured**2 + X_measured**2)
    Z_model = np.sqrt(R_model**2 + X_model**2)
    log_Z_measured = np.log10(Z_measured)
    slope_Z, intercept_Z, r_value_Z, _, _ = stats.linregress(log_freq, log_Z_measured)

    # Create interpolation functions for exact fit through measured points
    if len(freq) >= 4:
        interp_R = interp1d(
            log_freq,
            log_R_measured,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_X = interp1d(
            log_freq,
            log_X_measured,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_Z = interp1d(
            log_freq,
            log_Z_measured,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
    else:
        interp_R = interp1d(
            log_freq,
            log_R_measured,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_X = interp1d(
            log_freq,
            log_X_measured,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_Z = interp1d(
            log_freq,
            log_Z_measured,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
    # Generate smooth curves
    freq_smooth = np.logspace(np.log10(min(freq) * 0.5), np.log10(max(freq) * 2), 1000)
    R_smooth, X_smooth = cole_cole_impedance(freq_smooth, C_inf, C_0, tau_0, alpha)
    Z_smooth = np.sqrt(R_smooth**2 + X_smooth**2)


    # Generate interpolated curves that pass exactly through measured points
    freq_interp = np.logspace(np.log10(min(freq)), np.log10(max(freq)), 500)
    R_interp = 10 ** interp_R(np.log10(freq_interp))
    X_interp = 10 ** interp_X(np.log10(freq_interp))
    Z_interp = 10 ** interp_Z(np.log10(freq_interp))


    # Calculate local slopes at each measured point
    local_slopes_R = np.gradient(log_R_measured, log_freq)
    local_slopes_X = np.gradient(log_X_measured, log_freq)
    local_slopes_Z = np.gradient(log_Z_measured, log_freq)

    # Create figure with two subplots (like Cole Cole Parameters.py)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: R vs frequency
    ax1.loglog(freq, R_measured, "bo", label="Measured", markersize=8, zorder=3)
    ax1.loglog(
        freq_smooth, R_smooth, "b-", label="Cole-Cole Model", linewidth=2, alpha=0.8
    )
    ax1.loglog(
        freq_interp,
        R_interp,
        "k--",
        label=f"Interpolated Fit (avg slope={slope_R:.3f})",
        linewidth=1.5,
        alpha=0.7,
    )

    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Resistance (Ω)", fontsize=12)
    ax1.set_title("Resistance vs Frequency", fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend(fontsize=10)

    # Add text box with slope information
    slope_text = f"Average Slope: {slope_R:.3f}\nR²: {r_value_R**2:.3f}\n"
    slope_text += f"Local slopes: {local_slopes_R[0]:.3f} to {local_slopes_R[-1]:.3f}"
    ax1.text(
        0.05,
        0.95,
        slope_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Plot 2: |X| vs frequency
    ax2.loglog(freq, X_measured, "ro", label="Measured", markersize=8, zorder=3)
    ax2.loglog(
        freq_smooth, X_smooth, "r-", label="Cole-Cole Model", linewidth=2, alpha=0.8
    )
    ax2.loglog(
        freq_interp,
        X_interp,
        "k--",
        label=f"Interpolated Fit (avg slope={slope_X:.3f})",
        linewidth=1.5,
        alpha=0.7,
    )

    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel("|Reactance| (Ω)", fontsize=12)
    ax2.set_title("Reactance vs Frequency", fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend(fontsize=10)

    # Add text box with slope information
    slope_text = f"Average Slope: {slope_X:.3f}\nR²: {r_value_X**2:.3f}\n"
    slope_text += f"Local slopes: {local_slopes_X[0]:.3f} to {local_slopes_X[-1]:.3f}"
    ax2.text(
        0.05,
        0.95,
        slope_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.suptitle(
        f"Impedance vs Frequency - Cole-Cole Model Fit\n"
        + f"C_∞={C_inf*1e9:.3f} nF, C_0={C_0*1e9:.3f} nF, "
        + f"τ_0={tau_0*1e6:.1f} μs, α={alpha:.3f}",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Impedance_vs_Frequency_separate.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create log-log plot for |Z| vs frequency
    fig_z, ax_z = plt.subplots(figsize=(10, 8))
    
    # Plot measured |Z|
    ax_z.loglog(freq, Z_measured, "ko", label="Measured |Z|", markersize=10, 
                markerfacecolor="black", markeredgewidth=2, zorder=3)
    
    # Plot Cole-Cole model |Z|
    ax_z.loglog(freq_smooth, Z_smooth, "g-", label="Cole-Cole Model |Z|", 
                linewidth=3, alpha=0.8)
    
    # Plot interpolated fit
    ax_z.loglog(freq_interp, Z_interp, "k--",
                label=f"Interpolated Fit (avg slope={slope_Z:.3f})",
                linewidth=1.5, alpha=0.7)
    
    # Add reference line for average slope
    ref_line_freq = np.array([min(freq)*0.7, max(freq)*1.3])
    ref_line_Z = 10**(intercept_Z + slope_Z * np.log10(ref_line_freq))
    ax_z.loglog(ref_line_freq, ref_line_Z, "m:", 
                label=f"Slope reference: {slope_Z:.3f}", linewidth=1.5, alpha=0.5)
    
    # Mark characteristic frequency
    f_characteristic = 1 / (2 * np.pi * tau_0)
    ax_z.axvline(x=f_characteristic, color="red", linestyle=":", linewidth=2,
                 label=f"f₀ = {f_characteristic:.1f} Hz", alpha=0.7)
    
    # Formatting
    ax_z.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax_z.set_ylabel("|Z| (Ω)", fontsize=14, fontweight="bold")
    ax_z.set_title("Log |Z| vs Log f - Impedance Magnitude", fontsize=16, fontweight="bold")
    ax_z.grid(True, which="both", ls="-", alpha=0.3)
    ax_z.legend(fontsize=12, loc="best")
    
    # Add text box with analysis
    slope_text_z = f"Average Slope: {slope_Z:.3f}\nR²: {r_value_Z**2:.3f}\n"
    slope_text_z += f"Local slopes: {local_slopes_Z[0]:.3f} to {local_slopes_Z[-1]:.3f}\n\n"
    slope_text_z += f"Cole-Cole Parameters:\n"
    slope_text_z += f"C∞ = {C_inf*1e9:.2f} nF\n"
    slope_text_z += f"C₀ = {C_0*1e9:.2f} nF\n"
    slope_text_z += f"τ₀ = {tau_0*1e6:.1f} μs\n"
    slope_text_z += f"α = {alpha:.3f}"
    
    ax_z.text(0.05, 0.05, slope_text_z, transform=ax_z.transAxes, fontsize=11,
              verticalalignment="bottom", horizontalalignment="left",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Log_Z_vs_Log_f.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()

    # Create combined impedance plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.loglog(freq, R_measured, "bo", label="R measured", markersize=8)
    ax.loglog(freq, X_measured, "ro", label="|X| measured", markersize=8)
    ax.loglog(freq, Z_measured, "ko", label="|Z| measured", markersize=8)
    ax.loglog(freq_smooth, R_smooth, "b-", label="R Cole-Cole", linewidth=2)
    ax.loglog(freq_smooth, X_smooth, "r-", label="|X| Cole-Cole", linewidth=2)
    ax.loglog(freq_smooth, Z_smooth, "k-", label="|Z| Cole-Cole", linewidth=2)
    # Mark characteristic frequency
    ax.axvline(
        x=1 / (2 * np.pi * tau_0),
        color="g",
        linestyle=":",
        alpha=0.5,
        label=f"f_0 = {1/(2*np.pi*tau_0):.2f} Hz",
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Impedance (Ω)", fontsize=12)
    ax.set_title("Combined Impedance vs Frequency", fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Impedance_combined.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create residual analysis plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Resistance residuals
    R_residual_pct = (R_model - R_measured) / R_measured * 100
    ax1.semilogx(freq, R_residual_pct, "bo-", markersize=8)
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.5)
    ax1.set_ylabel("R Residual (%)", fontsize=12)
    ax1.set_title("Model Fit Residuals", fontsize=14)
    ax1.grid(True, which="both", alpha=0.3)

    # Reactance residuals
    X_residual_pct = (X_model - X_measured) / X_measured * 100
    ax2.semilogx(freq, X_residual_pct, "ro-", markersize=8)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel("|X| Residual (%)", fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "Impedance_residuals.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_impedance_results(
    freq, R_measured, X_measured, R_model, X_model, C_inf, C_0, tau_0, alpha, output_dir
):
    """
    Save impedance data and model parameters.
    Following the format of Cole Cole Parameters.py
    """
    # Calculate |Z|
    Z_measured = np.sqrt(R_measured**2 + X_measured**2)
    Z_model = np.sqrt(R_model**2 + X_model**2)

    # Save impedance data
    impedance_data = pd.DataFrame(
        {
            "Frequency_Hz": freq,
            "R_measured_Ohm": R_measured,
            "X_measured_Ohm": X_measured,
            "Z_measured_Ohm": Z_measured,
            "R_model_Ohm": R_model,
            "X_model_Ohm": X_model,
            "Z_model_Ohm": Z_model,
            "R_error_percent": (R_model - R_measured) / R_measured * 100,
            "X_error_percent": (X_model - X_measured) / X_measured * 100,
            "Z_error_percent": (Z_model - Z_measured) / Z_measured * 100,
            "Local_Slope_R": np.gradient(np.log10(R_measured), np.log10(freq)),
            "Local_Slope_X": np.gradient(np.log10(X_measured), np.log10(freq)),
            "Local_Slope_Z": np.gradient(np.log10(Z_measured), np.log10(freq)),
        }
    )

    impedance_data.to_csv(
        os.path.join(output_dir, "Model_Impedance_Plots.csv"), index=False
    )

    # Save interpolated data for smooth plots
    freq_interp = np.logspace(np.log10(min(freq) * 0.1), np.log10(max(freq) * 10), 500)
    R_interp, X_interp = cole_cole_impedance(freq_interp, C_inf, C_0, tau_0, alpha)
    Z_interp = np.sqrt(R_interp**2 + X_interp**2)

    interp_data = pd.DataFrame(
        {
            "Frequency_Hz": freq_interp,
            "R_interpolated_Ohm": R_interp,
            "X_interpolated_Ohm": X_interp,
            "Z_interpolated_Ohm": Z_interp,
        }
    )

    interp_data.to_csv(
        os.path.join(output_dir, "Model_Impedance_Interpolated_Fit.csv"), index=False
    )

    # Save Cole-Cole parameters
    params_data = pd.DataFrame(
        {
            "Parameter": [
                "C_inf (nF)",
                "C_0 (nF)",
                "tau_0 (us)",
                "alpha",
                "f_0 (Hz)",
                "Delta_epsilon_r",
            ],
            "Value": [
                C_inf * 1e9,
                C_0 * 1e9,
                tau_0 * 1e6,
                alpha,
                1 / (2 * np.pi * tau_0),
                (C_0 / C_inf - 1),
            ],
        }
    )

    params_data.to_csv(
        os.path.join(output_dir, "Cole_Cole_Parameters.csv"), index=False
    )

    print(f"\nSeparate impedance plots created with exact fit:")
    print(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    # Extract Cole-Cole parameters from validated impedance data
    # First try with balanced fitting
    print("=== BALANCED FITTING ===")
    params = fit_cole_cole_to_validated_impedance(
        csv_file="Impedance Plots Validation.csv",
        output_dir="cole_cole_impedance_results",
        prioritize_R=False,
    )

    # If R fit is poor, try again prioritizing R
    print("\n\n=== FITTING WITH R PRIORITIZED ===")
    params_R = fit_cole_cole_to_validated_impedance(
        csv_file="Impedance Plots Validation.csv",
        output_dir="cole_cole_impedance_results_R_priority",
        prioritize_R=True,
    )
