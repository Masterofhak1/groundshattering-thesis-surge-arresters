# Surge Arrester Testing - Analysis & Modeling

Hi! If you have found this repository for surge arrester testing (or stumbled by accident?), I would recommend you read first my thesis here: [Characterisation of Arresters for Harmonic Overvoltage Studies](https://resolver.tudelft.nl/uuid:81ae282d-0ad8-44b0-adf3-3be180313855) to understand the context behind most of the code(s).

To my supervisors visiting this page: send me an email when you're working on it ;)

---

## Overview

This repository contains the complete workflow for processing, analyzing, and modeling surge arresters in the insulation region. The procedure for obtaining results is fairly straightforward, as long as you keep track of the results (and the blinding amount of information in some of the intermediate Excel files). I'm happy to say it's not rocket science!

---

## Prerequisites

Before running the code, ensure you have the following installed:

### 1. Package Manager & Environment
Install **Conda** (recommended): [Installation Guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#regular-installation)

### 2. Code Editor
Install **VS Code**: [Download Here](https://code.visualstudio.com/)

### 3. Python Packages
Create a conda environment and install required packages:
```bash
conda create -n surge-arrester python=3.9
conda activate surge-arrester
conda install numpy pandas matplotlib scipy numba
```

---

## Code Documentation

All Python scripts in this repository are thoroughly documented with:
- Detailed docstrings for functions and classes
- Inline comments explaining key calculations and algorithms
- Parameter descriptions and units
- Usage examples in the main execution blocks

The scripts are designed to be self-explanatory - read through the code to understand the implementation details, algorithms, and any specific configuration options beyond what is documented in this README.

---

## Repository Structure

```
groundshattering-thesis-surge-arresters/
├── source codes/                    # Analysis scripts (see workflow below)
│   ├── A5 Analysis.py              # Step 1: Main AC/DC analysis
│   ├── A5 Energy and Power Analysis.py  # Step 2: Energy/power calculations
│   ├── nonlinear_resistive_current.py   # Step 3: Extract nonlinear V-I
│   ├── Cole Cole Model for Measurements.py  # Step 4: Cole-Cole parameter fitting
│   ├── Realising RC Networks.py    # Step 5: RC network synthesis
│   └── improved_ac_results.csv     # Example: Intermediate results file
├── examples/                        # Example datasets and results
├── data/                           # Input data files (scope measurements)
├── results/                        # Output files and plots
└── README.md                       # This file
```

---

## Complete Workflow - Step by Step

### Step 1: Data Collection from Oscilloscope

**What you do:**
1. Save data from your Picoscope (or similar oscilloscope)
2. Data format: 3 columns in CSV format
   - Column 1: Time (ms)
   - Column 2 (CH1): Voltage (kV)
   - Column 3 (CH3): Current (mA)
3. File naming convention: `A5_1_<frequency>-<number>.csv`
   - Example: `A5_1_17-0016.csv` (17 Hz, measurement #16)
   - Example: `A5_1_50-0016.csv` (50 Hz, measurement #16)
   - Example: `A5_1_DC-0010.csv` (DC measurement #10)

**Tip:** Keep the same measurement number across different frequencies for the same voltage level to make data tracking easier.

---

### Step 2: Run A5 Analysis Code

**Script:** [`source codes/A5 Analysis.py`](source%20codes/A5%20Analysis.py)

**What it does:**
- Processes oscilloscope data files
- Separates capacitive and resistive current components
- Calculates linear and nonlinear resistance components
- Generates waveform plots and comprehensive analysis

**Configuration (modify these lines before running):**

| Line | Parameter | Default Value | Description |
|------|-----------|---------------|-------------|
| **Line 2028** | `file_pattern` | `"A5_1_*csv"` | Pattern to match your data files |
| **Line 2028** | `output_dir` | `"results"` | Output folder for results |
| Line 389 | Voltage multiplier | `* 1000 * 3` | Amplifier gain (3x in this case) |

**How to run:**
```bash
# 1. Place your CSV data files in the same directory as the script
# 2. Update line 2028 if your files have a different pattern
# 3. Run the script
python "source codes/A5 Analysis.py"
```

**Outputs:**
- `results/improved_ac_results.csv` **IMPORTANT - needed for next steps**
- `results/improved_dc_results.csv` (if DC measurements exist)
- `results/<frequency>Hz/` folders with individual waveform plots and data
- Various plots: capacitance, resistance, V-I characteristics

**Critical:** The `improved_ac_results.csv` file is the key output - you'll need it for Step 3.

---

### Step 2b: Run Energy and Power Analysis

**Script:** [`source codes/A5 Energy and Power Analysis.py`](source%20codes/A5%20Energy%20and%20Power%20Analysis.py)

**What it does:**
- Calculates energy dissipation and power over time
- Takes approximately **35 minutes** to run
- Creates detailed energy/power time series plots
- Required for final model validation and comparison between measured arrester and synthesized model

**Configuration:**

| Line | Parameter | Default Value | Description |
|------|-----------|---------------|-------------|
| **Line 1038** | `file_pattern` | `"A5_1_*.csv"` | Input file pattern |
| **Line 1039** | `output_dir` | `"energy_results_peak_start"` | Output directory |
| Line 1040 | `start_from_first_peak` | `True` | Start analysis from first voltage peak |
| Line 1041 | `skip_initial_samples` | `10` | Skip initial transient samples |

**How to run:**
```bash
python "source codes/A5 Energy and Power Analysis.py"
```

**Outputs:**
- Energy and power time series CSV files
- Comparative plots across frequencies
- Summary CSV with energy metrics

---

### Step 3: Extract Nonlinear V-I Characteristics

**Script:** [`source codes/nonlinear_resistive_current.py`](source%20codes/nonlinear_resistive_current.py)

**What it does:**
- Takes the `improved_ac_results.csv` from Step 2
- Calculates nonlinear resistive current component
- Creates polynomial fit for ATP/EMTP modeling
- Generates Type 92 element table for power system simulations

**Prerequisites:**
1. Copy `results/improved_ac_results.csv` to the same directory as this script
2. The script expects this file to be in the current directory

**Configuration:**

| Line | Parameter | Default Value | Description |
|------|-----------|---------------|-------------|
| **Line 524** | `input_file` | `"improved_ac_results.csv"` | Input filename |
| **Line 525** | `output_dir` | `"Nonlinear_resistive_current_results"` | Output folder |
| Line 526 | `nonlinear_res_threshold` | `900` MΩ | Threshold for filtering high resistance values |
| Line 527 | `type92_points` | `20` | Number of points for Type 92 table |

**How to run:**
```bash
# 1. Copy improved_ac_results.csv to this script's directory
cp results/improved_ac_results.csv .

# 2. Run the script
python "source codes/nonlinear_resistive_current.py"
```

**Outputs:**
- `Nonlinear_resistive_current_results/calculated_nonlinear_vi.csv`
- `Nonlinear_resistive_current_results/type92_element_table.csv` **For ATP/EMTP modeling**
- `Nonlinear_resistive_current_results/type92_element_table.txt` (formatted table)
- Frequency-based V-I curves and polynomial fit plots

**Important:** The `type92_element_table.csv` contains the nonlinear V-I data for ATP/EMTP Type 92 surge arrester modeling.

---

### Step 4: Manual Data Extraction for Cole-Cole Fitting

**What you do manually:**
For a specific voltage level across different frequencies, you need to extract R_linear and Cs values to calculate equivalent impedance parameters.

**Example:** For voltage point #16 across frequencies:
- From `improved_ac_results.csv`, find rows for:
  - `A5_1_17-0016` → R_linear, Cs at 17 Hz
  - `A5_1_50-0016` → R_linear, Cs at 50 Hz
  - `A5_1_150-0016` → R_linear, Cs at 150 Hz
  - etc.

**Calculations to perform:**
1. `|X_C| = 1 / (2π × f × Cs)` - Capacitive reactance magnitude
2. `R_eq = R_linear` - Equivalent resistance
3. `|X_eq| = |X_C|` - Equivalent reactance magnitude

**Create Excel/CSV file:**
Create a file named `Impedance Plots Validation.csv` (or similar) with columns:
- Frequency (Hz)
- R_eq (Ω or MΩ)
- |X_eq| (Ω or MΩ)
- Voltage (kV)

**Example data in:** [`source codes/Impedance Plots Validation.xlsx`](source%20codes/Impedance%20Plots%20Validation.xlsx) and [`source codes/Impedance Plots Validation.csv`](source%20codes/Impedance%20Plots%20Validation.csv)

---

### Step 5: Cole-Cole Model Parameter Fitting

**Script:** [`source codes/Cole Cole Model for Measurements.py`](source%20codes/Cole%20Cole%20Model%20for%20Measurements.py)

**What it does:**
- Fits Cole-Cole dielectric model to your impedance measurements
- Extracts characteristic parameters (C_inf, C_0, tau_0, alpha)
- Generates two versions: balanced fit and R-prioritized fit

**Prerequisites:**
1. Create `Impedance Plots Validation.csv` from Step 4
2. Place it in the same directory as this script

**Configuration:**

| Line | Parameter | Default Value | Description |
|------|-----------|---------------|-------------|
| **Line 733** | `csv_file` | `"Impedance Plots Validation.csv"` | Input impedance data file |
| **Line 734** | `output_dir` | `"cole_cole_impedance_results"` | Output folder (balanced fit) |
| **Line 742** | `output_dir` | `"cole_cole_impedance_results_R_priority"` | Output folder (R-prioritized) |

**How to run:**
```bash
# 1. Ensure Impedance Plots Validation.csv is in the same directory
# 2. Run the script (it runs both fitting methods automatically)
python "source codes/Cole Cole Model for Measurements.py"
```

**Outputs:**
- `cole_cole_impedance_results/Cole_Cole_Parameters.csv` (balanced fit)
- `cole_cole_impedance_results_R_priority/Cole_Cole_Parameters.csv` **(recommended)**
- Impedance plots comparing measured vs fitted data

**Tip:** Use the **R_priority** folder results - they typically provide better fits for resistive characteristics.

---

### Step 6: RC Network Synthesis

**Script:** [`source codes/Realising RC Networks.py`](source%20codes/Realising%20RC%20Networks.py)

**What it does:**
- Synthesizes Foster I and Foster II RC network approximations
- Uses Oustaloup fractional-order approximation method
- Generates network topology diagrams and frequency response comparisons

**Prerequisites:**
1. Get Cole-Cole parameters from `Cole_Cole_Parameters.csv` (Step 5)
2. Manually edit the script to input these parameters

**Configuration:**

You **MUST manually edit** lines 717-725 with your Cole-Cole parameters:

| Line | Parameter | Example Value | Description |
|------|-----------|---------------|-------------|
| **Line 718** | `f0` | `1` Hz | Minimum frequency for RC model operation |
| **Line 719** | `f1` | `1000.0` Hz | Maximum frequency for RC model operation |
| **Line 720** | `N` | `3` | Number of RC branches |
| **Line 721** | `alpha` | `0.587` | From Cole_Cole_Parameters.csv |
| **Line 722** | `tau0` | `0.1` s | From Cole_Cole_Parameters.csv (tau_0 × 10^-6) |
| **Line 723** | `C_inf` | `0.521e-9` F | From Cole_Cole_Parameters.csv (C_inf × 10^-9) |
| **Line 724** | `C0` | `0.882e-9` F | From Cole_Cole_Parameters.csv (C_0 × 10^-9) |

**Important Notes:**
- **Frequency range (f0, f1):** Extend by 1-2 decades on each side of your measurement range to avoid approximation errors at corner frequencies
  - Example: If you measured 10-500 Hz, use f0=1 Hz and f1=5000 Hz
- **Number of branches (N):** More branches = better approximation but more complex circuit
  - Start with N=3, increase if needed

**How to run:**
```bash
# 1. Open the script in a text editor
# 2. Edit lines 717-725 with parameters from Cole_Cole_Parameters.csv
# 3. Save and run
python "source codes/Realising RC Networks.py"
```

**Outputs:**
- `oustaloup_with_lines_and_labels.png` - Frequency response with pole/zero locations
- `foster_topologies_comparison.png` - Foster I vs Foster II network comparison
- `complete_capacitor_models.png` - Complete model frequency response
- Console output: R and C values for Foster I and Foster II networks

**Network topologies:**
- **Foster I:** Series RC branches in parallel
- **Foster II:** Parallel RC branches in series
- Choose based on your circuit simulation requirements

---

## Quick Reference: File Dependencies

```
Scope Data (CSV)
    ↓
[Step 1] A5 Analysis.py
    ↓
    ├→ improved_ac_results.csv ──┐
    ↓                             ↓
[Step 2] nonlinear_resistive_current.py
    ↓
    └→ type92_element_table.csv (for ATP/EMTP)

From improved_ac_results.csv:
    ↓
[Step 3] Manual extraction → Impedance Plots Validation.csv
    ↓
[Step 4] Cole Cole Model for Measurements.py
    ↓
    └→ Cole_Cole_Parameters.csv
        ↓
[Step 5] Realising RC Networks.py → Foster I/II circuits
```

---

## Common Issues & Troubleshooting

### File Not Found Errors
- **Problem:** `improved_ac_results.csv` not found
- **Solution:** Make sure you've run A5 Analysis.py first and copy the file to the script directory

### Import Errors
- **Problem:** `ModuleNotFoundError: No module named 'numba'`
- **Solution:** Install missing packages:
  ```bash
  conda install numba numpy pandas matplotlib scipy
  ```

### Memory Issues (Energy Analysis)
- **Problem:** Script runs very slow or crashes
- **Solution:** The energy analysis processes large datasets. Ensure you have:
  - At least 8GB RAM available
  - Close other applications
  - Process fewer files at once

### Wrong Results
- **Problem:** Unexpected parameter values
- **Solution:** Check your input data:
  - Verify voltage amplifier gain (line 389 in A5 Analysis.py)
  - Check file naming convention matches pattern
  - Ensure CSV format is correct (3 columns)

---

## Tips for Success

1. **Keep organized:** Create separate folders for each voltage/frequency sweep
2. **Consistent naming:** Use the same file number for same voltage across frequencies
   - Good: `A5_1_17-0016.csv`, `A5_1_50-0016.csv`, `A5_1_150-0016.csv`
   - Bad: `A5_1_17-0016.csv`, `A5_1_50-0023.csv`, `A5_1_150-0008.csv`
3. **Track intermediate files:** The `improved_ac_results.csv` is crucial - don't delete it!
4. **Backup your data:** Copy scope data files before processing
5. **Check units:** Be careful with conversions (kV ↔ V, mA ↔ A, nF ↔ F)

---

## Citation

If you use this code or methodology in your research, please cite:
```
https://resolver.tudelft.nl/uuid:81ae282d-0ad8-44b0-adf3-3be180313855
```

---

## Contact

Questions or issues? Feel free to reach out!

---

*Last updated: November 2025*
