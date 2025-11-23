import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_improved_ac_data(filename, nonlinear_res_threshold=900):
    """
    Load data from improved_ac_results.csv and calculate non-linear current
    
    Parameters:
    -----------
    filename : str
        Path to improved_ac_results.csv file
    nonlinear_res_threshold : float
        Non-linear resistance threshold in megaohms. Values above this will be excluded.
    
    Returns:
    --------
    DataFrame with Voltage (kV), Nonlinear Current (mA), and Frequency (Hz) columns
    """
    # Load the CSV file
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} data points from {filename}")
    
    # Display column names for verification
    print(f"Available columns: {', '.join(df.columns)}")
    
    # Apply non-linear resistance threshold filter (Column U - R_nonlinear)
    if 'R_nonlinear (MΩ)' in df.columns:
        original_count = len(df)
        df = df[df['R_nonlinear (MΩ)'] <= nonlinear_res_threshold]
        filtered_count = len(df)
        print(f"Applied non-linear resistance threshold of {nonlinear_res_threshold} MΩ:")
        print(f"Filtered out {original_count - filtered_count} points, {filtered_count} points remaining")
    elif 'R_nonlinear' in df.columns:
        # If the column is in ohms instead of megaohms
        original_count = len(df)
        df = df[df['R_nonlinear'] <= nonlinear_res_threshold * 1e6]  # Convert MΩ to Ω
        filtered_count = len(df)
        print(f"Applied non-linear resistance threshold of {nonlinear_res_threshold} MΩ:")
        print(f"Filtered out {original_count - filtered_count} points, {filtered_count} points remaining")
    
    # Extract voltage (peak), non-linear resistance, and frequency
    v_peak = df['v_peak'].values  # Column D
    r_nonlinear = df['R_nonlinear'].values  # Column U
    
    # Get frequency information - try different possible column names
    if 'standardized_freq' in df.columns:
        frequency = df['standardized_freq'].values
    else:
        print("Warning: No frequency column found. Using default value of 50 Hz.")
        frequency = np.ones_like(v_peak) * 50  # Default to 50 Hz
    
    # If R_nonlinear (MΩ) exists, use that instead for unit consistency
    if 'R_nonlinear (MΩ)' in df.columns:
        r_nonlinear = df['R_nonlinear (MΩ)'].values * 1e6  # Convert MΩ to Ω
    
    # Calculate non-linear current using Ohm's law: I = V/R
    # Convert to mA (assuming v_peak is in V and r_nonlinear is in Ω)
    nonlinear_current_ma = (v_peak / r_nonlinear) * 1000
    
    # Convert voltage to kV
    voltage_kv = v_peak / 1000
    
    # Calculate resistance in MΩ for output table
    nonlinear_resistance_mohm = r_nonlinear / 1e6
    
    # Create new dataframe with required columns
    result_df = pd.DataFrame({
        'Voltage (kV)': voltage_kv,
        'Nonlinear Current (mA)': nonlinear_current_ma,
        'Nonlinear Resistance (MΩ)': nonlinear_resistance_mohm,
        'Frequency (Hz)': frequency
    })
    
    # Filter out any invalid data points (negative or zero values)
    result_df = result_df[(result_df['Voltage (kV)'] > 0) & 
                          (result_df['Nonlinear Current (mA)'] > 0)]
    
    # Sort by frequency and then by current for better visualization
    result_df = result_df.sort_values(['Frequency (Hz)', 'Nonlinear Current (mA)'])
    
    # Summary of calculated data
    print(f"\nData Summary after calculation:")
    print(f"Current range: {result_df['Nonlinear Current (mA)'].min():.6f} to {result_df['Nonlinear Current (mA)'].max():.6f} mA")
    print(f"Voltage range: {result_df['Voltage (kV)'].min():.6f} to {result_df['Voltage (kV)'].max():.6f} kV")
    print(f"Frequency range: {result_df['Frequency (Hz)'].min():.1f} to {result_df['Frequency (Hz)'].max():.1f} Hz")
    
    # Count unique frequencies
    unique_freqs = result_df['Frequency (Hz)'].unique()
    print(f"Number of different frequencies: {len(unique_freqs)}")
    print(f"Unique frequencies: {', '.join([f'{f:.1f} Hz' for f in sorted(unique_freqs)])}")
    
    return result_df


def polynomial_degree3_fit(df, output_dir="fitted_results"):
    """
    Polynomial fit (degree 3) in log-log space for curved V-I characteristics
    Creates both log-log and linear-log plots
    """
    print("\n=== Polynomial Degree 3 Fitting for Surge Arrester Nonlinear V-I Characteristics ===")

    # Extract data
    current = df["Nonlinear Current (mA)"].values
    voltage = df["Voltage (kV)"].values

    # Filter valid data
    mask = (current > 0) & (voltage > 0)
    current = current[mask]
    voltage = voltage[mask]

    # Convert to log space
    log_i = np.log10(current)
    log_v = np.log10(voltage)

    # Polynomial fit (degree 3)
    coeffs = np.polyfit(log_i, log_v, 3)
    poly = np.poly1d(coeffs)

    print(f"Polynomial coefficients (degree 3):")
    for i, c in enumerate(coeffs):
        print(f"  c{3-i}: {c:.6f}")

    # Generate fitted curve
    log_i_fit = np.linspace(log_i.min(), log_i.max(), 500)
    log_v_fit = poly(log_i_fit)

    i_fit = 10**log_i_fit
    v_fit = 10**log_v_fit

    # Ensure we don't exceed max voltage
    v_max = voltage.max()
    valid_mask = v_fit <= v_max
    i_fit = i_fit[valid_mask]
    v_fit = v_fit[valid_mask]

    # Calculate R-squared
    log_v_predicted = poly(log_i)
    v_predicted = 10**log_v_predicted
    ss_res = np.sum((voltage - v_predicted) ** 2)
    ss_tot = np.sum((voltage - np.mean(voltage)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.4f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Log-Log plot
    plt.figure(figsize=(12, 8))
    plt.scatter(current, voltage, c="blue", s=50, alpha=0.6, label="Measured V-I Data")
    plt.plot(i_fit, v_fit, "r-", linewidth=3, label=f"Polynomial (degree 3) fit")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(current.min() * 0.8, current.max() * 1.2)
    plt.ylim(voltage.min() * 0.8, voltage.max() * 1.2)

    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Nonlinear Resistive Current [mA]", fontsize=12)
    plt.ylabel("Voltage [kV]", fontsize=12)
    plt.title(
        "Surge Arrester Nonlinear V-I Characteristics (Log-Log Scale)", fontsize=14
    )
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/polynomial_fit_loglog.png", dpi=300)
    plt.close()  # Using close instead of show for non-interactive environments

    # Plot 2: Linear Voltage - Log Current plot
    plt.figure(figsize=(12, 8))
    plt.scatter(current, voltage, c="blue", s=50, alpha=0.6, label="Measured V-I Data")
    plt.plot(i_fit, v_fit, "r-", linewidth=3, label=f"Polynomial (degree 3) fit")

    plt.xscale("log")  # Log scale for current
    # y-axis remains linear for voltage
    plt.xlim(current.min() * 0.8, current.max() * 1.2)
    plt.ylim(0, voltage.max() * 1.1)

    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Nonlinear Resistive Current [mA]", fontsize=12)
    plt.ylabel("Voltage [kV]", fontsize=12)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/polynomial_fit_linear_log.png", dpi=300)
    plt.close()  # Using close instead of show for non-interactive environments

    # Save fitted data for Type 92 element
    fit_data = pd.DataFrame({"Current (mA)": i_fit, "Voltage (kV)": v_fit})
    fit_data.to_csv(f"{output_dir}/type92_polynomial_fit.csv", index=False)
    print(f"\nFitted data saved to {output_dir}/type92_polynomial_fit.csv")

    return coeffs, i_fit, v_fit


def create_type92_table(fit_file, output_dir="fitted_results", n_points=29):
    """
    Create a formatted table for Type 92 element input for surge arrester modeling
    
    Parameters:
    -----------
    fit_file : str
        Path to the polynomial fit CSV file
    output_dir : str
        Directory to save output files
    n_points : int
        Number of points for Type 92 element (default is 29 as per requirement)
    """
    print(f"\n=== Creating Type 92 Element Table for Surge Arrester Modeling ({n_points} points) ===")

    # Load fitted data
    fit_data = pd.read_csv(fit_file)

    # Select evenly spaced points in log scale for better representation of surge arrester behavior
    # Use exactly n_points to span the entire range (29 points for Type 92)
    log_currents = np.log10(fit_data["Current (mA)"])
    indices = np.linspace(0, len(fit_data) - 1, n_points, dtype=int)

    # Create table
    table = pd.DataFrame(
        {
            "Point": range(1, n_points + 1),
            "Current (A)": fit_data.iloc[indices]["Current (mA)"].values / 1000,  # Convert mA to A
            "Voltage (V)": fit_data.iloc[indices]["Voltage (kV)"].values * 1000,  # Convert kV to V
        }
    )
    
    # Verify current is in Amperes (should be very small values)
    min_current_a = table["Current (A)"].min()
    max_current_a = table["Current (A)"].max()
    print(f"Current range in Type 92 table: {min_current_a:.6e} A to {max_current_a:.6e} A")

    # Calculate resistance at each point (useful for surge arrester modeling)
    # Using plain "Ohm" instead of the Greek Omega symbol
    table["Resistance (Ohm)"] = table["Voltage (V)"] / table["Current (A)"]

    # Make sure Point is integer type
    table["Point"] = table["Point"].astype(int)

    # Save as formatted text file - using explicit ASCII encoding to avoid any issues
    try:
        with open(f"{output_dir}/type92_element_table.txt", "w", encoding="ascii") as f:
            f.write(f"Type 92 Nonlinear Resistor Element Table for Surge Arrester Modeling ({n_points} points)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Point':>5} {'Current (A)':>12} {'Voltage (V)':>12} {'Resistance (Ohm)':>15}\n")
            f.write("-" * 50 + "\n")

            for _, row in table.iterrows():
                point_val = int(row["Point"])
                f.write(
                    f"{point_val:5d} {row['Current (A)']:12.6e} {row['Voltage (V)']:12.1f} {row['Resistance (Ohm)']:15.2f}\n"
                )
        print(f"Text file saved: {output_dir}/type92_element_table.txt")
    except Exception as e:
        print(f"Error saving text file: {e}")
        print("Trying alternative method...")
        # Alternative method without f-strings
        with open(f"{output_dir}/type92_element_table.txt", "w") as f:
            f.write(f"Type 92 Nonlinear Resistor Element Table for Surge Arrester Modeling ({n_points} points)\n")
            f.write("=" * 70 + "\n\n")
            f.write("Point  Current (A)   Voltage (V)   Resistance (Ohm)\n")
            f.write("-" * 50 + "\n")
            
            for _, row in table.iterrows():
                point_val = int(row["Point"])
                current = row["Current (A)"]
                voltage = row["Voltage (V)"]
                resistance = row["Resistance (Ohm)"]
                line = "{:5d} {:12.6e} {:12.1f} {:15.2f}\n".format(point_val, current, voltage, resistance)
                f.write(line)
        print(f"Text file saved using alternative method")

    # Also save as CSV (using plain ASCII to avoid any issues)
    try:
        table.to_csv(f"{output_dir}/type92_element_table.csv", index=False, encoding="ascii")
        print(f"CSV file saved: {output_dir}/type92_element_table.csv")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        # Create a simple copy without any special characters
        simple_table = pd.DataFrame({
            "Point": table["Point"],
            "Current_A": table["Current (A)"],
            "Voltage_V": table["Voltage (V)"],
            "Resistance_Ohm": table["Resistance (Ohm)"]
        })
        simple_table.to_csv(f"{output_dir}/type92_element_table_simple.csv", index=False)
        print(f"Simplified CSV file saved")

    print(f"Type 92 table saved with {n_points} points spanning the entire range of the polynomial fit")
    
    # Display first 10 rows
    print("\nFirst 10 rows of Type 92 table for surge arrester model:")
    print(table.head(10).to_string())
    
    # Display the last 2 rows to show the full range
    print("\nLast 2 rows (showing maximum values):")
    print(table.tail(2).to_string())

    return table


def create_frequency_based_table(df, output_dir="fitted_results"):
    """
    Create a frequency-based table of voltage, current, and resistance data
    with color coding for different frequencies
    """
    print(f"\n=== Creating Frequency-Based V-I-R Table for Surge Arrester ===")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by frequency
    freq_groups = df.groupby('Frequency (Hz)')
    
    # Get unique frequencies
    unique_freqs = sorted(df['Frequency (Hz)'].unique())
    
    # Prepare HTML table with color coding
    html_output = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Surge Arrester V-I-R Data by Frequency</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333366; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th { background-color: #333366; color: white; padding: 10px; text-align: left; }
            td { padding: 8px; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .freq-header { text-align: center; font-weight: bold; padding: 10px; color: white; }
            .summary { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Surge Arrester V-I-R Characteristics by Frequency</h1>
        <p>Color-coded data showing surge arrester behavior at different test frequencies.</p>
        
        <table border="1">
            <tr>
                <th>Frequency (Hz)</th>
                <th>Voltage (kV)</th>
                <th>Nonlinear Current (mA)</th>
                <th>Nonlinear Resistance (MOhm)</th>
            </tr>
    """
    
    # Color palette for different frequencies (up to 10 distinct colors)
    colors = [
        "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
        "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"
    ]
    
    # Generate HTML rows with color-coding by frequency
    for i, freq in enumerate(unique_freqs):
        color_idx = i % len(colors)
        color = colors[color_idx]
        
        # Get data for this frequency
        freq_data = freq_groups.get_group(freq)
        
        # Add a header row for this frequency
        html_output += f"""
            <tr>
                <td colspan="4" class="freq-header" style="background-color: {color};">
                    {freq} Hz ({len(freq_data)} data points)
                </td>
            </tr>
        """
        
        # Add data rows
        for _, row in freq_data.iterrows():
            html_output += f"""
            <tr>
                <td>{row['Frequency (Hz)']}</td>
                <td>{row['Voltage (kV)']:.4f}</td>
                <td>{row['Nonlinear Current (mA)']:.6f}</td>
                <td>{row['Nonlinear Resistance (MΩ)']:.4f}</td>
            </tr>
            """
    
    # Close HTML table and document
    html_output += """
        </table>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>This table shows the voltage-current-resistance characteristics of the surge arrester measured at different frequencies.</p>
            <p>Each color represents a different test frequency, demonstrating the frequency-dependent behavior of the arrester.</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file with explicit UTF-8 encoding
    try:
        html_path = os.path.join(output_dir, "frequency_based_vi_table.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"HTML table saved with UTF-8 encoding: {html_path}")
    except Exception as e:
        print(f"Error saving HTML with UTF-8 encoding: {e}")
        
        # Try alternative method with ASCII only - replace any Unicode characters
        html_output = html_output.replace("MΩ", "MOhm")
        html_path = os.path.join(output_dir, "frequency_based_vi_table_ascii.html")
        with open(html_path, "w", encoding="ascii", errors="replace") as f:
            f.write(html_output)
        print(f"Fallback: HTML table saved with ASCII encoding: {html_path}")
    
    # Also create a CSV file (grouped by frequency)
    csv_path = os.path.join(output_dir, "frequency_based_vi_table.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Frequency-based data also saved as CSV: {csv_path}")
    
    # Create a summary of data points per frequency
    freq_summary = df.groupby('Frequency (Hz)').size().reset_index(name='Data Points')
    freq_summary.columns = ['Frequency (Hz)', 'Number of Data Points']
    freq_summary_path = os.path.join(output_dir, "frequency_summary.csv")
    freq_summary.to_csv(freq_summary_path, index=False)
    print(f"Frequency summary saved: {freq_summary_path}")
    
    return freq_summary


def plot_frequency_based_vi_curves(df, output_dir="fitted_results"):
    """
    Create color-coded V-I curves based on frequency
    """
    print(f"\n=== Creating Frequency-Based V-I Curves for Surge Arrester ===")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique frequencies
    unique_freqs = sorted(df['Frequency (Hz)'].unique())
    
    # Color palette for different frequencies (up to 10 distinct colors)
    colors = [
        "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
        "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"
    ]
    
    # Create Log-Log plot with color-coding by frequency
    plt.figure(figsize=(12, 8))
    
    for i, freq in enumerate(unique_freqs):
        color_idx = i % len(colors)
        color = colors[color_idx]
        
        # Filter data for this frequency
        freq_data = df[df['Frequency (Hz)'] == freq]
        
        # Plot the data
        plt.loglog(
            freq_data['Nonlinear Current (mA)'],
            freq_data['Voltage (kV)'],
            'o-',
            color=color,
            label=f"{freq} Hz",
            markersize=6,
            linewidth=2
        )
    
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Nonlinear Resistive Current [mA]", fontsize=12)
    plt.ylabel("Voltage [kV]", fontsize=12)
    plt.title("Surge Arrester V-I Characteristics by Frequency (Log-Log Scale)", fontsize=14)
    plt.legend(fontsize=10, title="Test Frequency")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/frequency_based_vi_curves_loglog.png", dpi=300)
    plt.close()
    
    # Create Linear-Log plot with color-coding by frequency
    plt.figure(figsize=(12, 8))
    
    for i, freq in enumerate(unique_freqs):
        color_idx = i % len(colors)
        color = colors[color_idx]
        
        # Filter data for this frequency
        freq_data = df[df['Frequency (Hz)'] == freq]
        
        # Plot the data
        plt.semilogx(
            freq_data['Nonlinear Current (mA)'],
            freq_data['Voltage (kV)'],
            'o-',
            color=color,
            label=f"{freq} Hz",
            markersize=6,
            linewidth=2
        )
    
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Nonlinear Resistive Current [mA]", fontsize=12)
    plt.ylabel("Voltage [kV]", fontsize=12)
    plt.title("Surge Arrester V-I Characteristics by Frequency (Linear-Log Scale)", fontsize=14)
    plt.legend(fontsize=10, title="Test Frequency")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/frequency_based_vi_curves_linearlog.png", dpi=300)
    plt.close()
    
    print(f"Frequency-based V-I curves saved as:")
    print(f"- {output_dir}/frequency_based_vi_curves_loglog.png")
    print(f"- {output_dir}/frequency_based_vi_curves_linearlog.png")
    
    return unique_freqs


def main():
    """
    Main function to process improved_ac_results.csv, calculate non-linear current,
    and perform polynomial degree 3 fitting for surge arrester V-I characteristics
    """
    # Configuration
    input_file = "improved_ac_results.csv"
    output_dir = "Nonlinear_resistive_current_results"
    nonlinear_res_threshold = 900  # MΩ threshold for non-linear resistance
    type92_points = 20  # Number of points for Type 92 element table (required exactly 29)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("SURGE ARRESTER NONLINEAR V-I CHARACTERISTIC ANALYSIS")
    print("=" * 80)
    
    # Load data and calculate non-linear current
    print(f"\nProcessing file: {input_file}")
    print(f"Using non-linear resistance threshold: {nonlinear_res_threshold} MΩ")
    df = load_improved_ac_data(input_file, nonlinear_res_threshold)
    
    # Save the calculated non-linear current data
    df.to_csv(f"{output_dir}/calculated_nonlinear_vi.csv", index=False)
    print(f"Calculated V-I data saved to {output_dir}/calculated_nonlinear_vi.csv")
    
    # Create frequency-based table with color coding
    create_frequency_based_table(df, output_dir)
    
    # Create frequency-based V-I curves
    plot_frequency_based_vi_curves(df, output_dir)

    print("\n" + "=" * 50)
    print("POLYNOMIAL DEGREE 3 FITTING FOR SURGE ARRESTER MODEL")
    print("=" * 50)

    # Run polynomial degree 3 fit
    coeffs, i_fit, v_fit = polynomial_degree3_fit(df, output_dir)

    # Create Type 92 element table for surge arrester modeling (using exactly 29 points)
    create_type92_table(f"{output_dir}/type92_polynomial_fit.csv", output_dir, type92_points)

    print("\n" + "=" * 50)
    print("SURGE ARRESTER ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"\nResults saved in '{output_dir}' directory:")
    print("- calculated_nonlinear_vi.csv (Calculated V-I data)")
    print("- frequency_based_vi_table.html (Color-coded frequency table)")
    print("- frequency_based_vi_table.csv (CSV frequency data)")
    print("- frequency_based_vi_curves_loglog.png (V-I curves by frequency)")
    print("- frequency_based_vi_curves_linearlog.png (V-I curves by frequency)")
    print("- polynomial_fit_loglog.png (Polynomial fit plot)")
    print("- polynomial_fit_linear_log.png (Polynomial fit plot)")
    print("- type92_polynomial_fit.csv (Fitted data)")
    print(f"- type92_element_table.txt (Formatted table for Type 92 with {type92_points} points)")
    print(f"- type92_element_table.csv (CSV table for Type 92 with {type92_points} points)")

    # Display polynomial equation for surge arrester model
    print("\n" + "=" * 50)
    print("SURGE ARRESTER POLYNOMIAL EQUATION (DEGREE 3)")
    print("=" * 50)
    print("log10(V) = c0 * log10(I)³ + c1 * log10(I)² + c2 * log10(I) + c3")
    print(f"where:")
    for i, c in enumerate(coeffs):
        print(f"  c{3-i} = {c:.6f}")

    print("\nThis polynomial fit provides the curved characteristic typical of")
    print("surge arresters and is suitable for Type 92 element modeling in")
    print("power system simulation software for transient and overvoltage studies.")


if __name__ == "__main__":
    main()