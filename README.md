# Characterisation of Arresters of Harmonic Overvoltage Studies

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

### Already have a conda environment?
It could be that these packages are already installed in your conda environment, but its always good to check twice.

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
│   ├── improved_ac_results.csv     # Example: Intermediate results file
│   ├── Impedance Plots Validation.csv   # Example: Input for Cole-Cole fitting
│   └── Impedance Plots Validation.xlsx  # Example: Input for Cole-Cole fitting
├── examples/                        # Example datasets and measurement tallies
│   ├── Measurements Tally A5.xlsx  # Example measurement tracking sheet
│   └── Measurement Tally.xlsx      # Example measurement tracking sheet
├── data/                           # Input data files (scope measurements)
├── results/                        # Output files and plots
├── modelling/                      # Model files (ATP/EMTP, etc.)
└── README.md                       # This file
```

---

## Complete Workflow - Step by Step

### Step 1: Data Collection from Oscilloscope

**What you do:**
1. Save data from your Picoscope (or similar oscilloscope). I usually keep track of the test cases using an excel sheet; to have an idea of the progress and everything in one place. See [examples here](examples/Measurements%20Tally%20A5.xlsx) & [here](examples/Measurement%20Tally.xlsx)
2. Data format: 3 columns in CSV format
   - Column 1: Time (ms)
   - Column 2 (CH1): Voltage (kV)
   - Column 3 (CH3): Current (mA)
3. File naming convention: `A5_1_<frequency>-<number>.csv` is recommended
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

#### Understanding the Configurable Parameters

The code is used to analyse the MOV data. I recommend you to save each file in a pattern, for the code to run smoothly (and not risk missing data). The following pointers explain the sections of the code that can be configured based on your preferences and test conditions, along with brief explanation of how varying these parameters can affect your results:

**1. Window Size (Smoothing)**
```python
window_size = int(round(0.0001 * Fs))
```
`0.0001` is the window size selected. If you *decrease* the window size to e.g. `0.001`, you're averaging fewer samples which would aggressively smoothen your waveforms and you could lose crucial waveform information (like peaks, transients and noise). Likewise if you *increase* the window size to `0.00001*Fs`, you're averaging across a larger number of samples.

You will need to adjust this rectangular window size based on the -3dB roll-off of your low pass filter. In the code, the LPF was 5kHz, hence the 100 µs size.

**2. Target Frequencies**
```python
target_freqs = [10, 17, 27, 50, 100, 150, 300, 500]
for target_freq in target_freqs:
    if abs(freq - target_freq) <= 1.5:
        return target_freq
return round(freq)
```
Target frequency is the preset frequency options in the code. If the analysis iteration detects a frequency of the signal near these frequencies, it assumes it is THAT frequency, and considers the further analysis using that frequency value. You may change the values/index here based on the frequencies you are testing the MOV.

Why? Often real world measurements are not precise. Side note - real arrester current also has simultaneously several harmonic currents, which is a limitation of this code. See Chapter 6 of thesis for more information.

**3. Dynamic Sampling Based on Frequency**
```python
# "ENHANCED" - Dynamic sampling based on frequency
# Quick frequency detection for sampling strategy
quick_samples = min(2**17, len(time_all_proc) - 1000)
quick_time = time_all_proc[1000:1000 + quick_samples]
quick_voltage = ch1_debiased[1000:1000 + quick_samples]

# Detect frequency
quick_freq, _, _, _ = iDFT3pHann(quick_voltage, quick_time[-1] - quick_time[0])
std_freq = standardize_frequency(quick_freq)

# Dynamic sampling
if std_freq == 10:
    start_idx = 1000
    num_samples = 2**19  # 524,288 samples for 10 Hz, because of long period.
    print(f"\n[10 Hz Mode] Extended sampling: {num_samples:,} samples")
else:
    start_idx = 1000
    num_samples = 2**17  # 131,072 samples for other frequencies
    print(f"\n[{std_freq} Hz Mode] Standard sampling: {num_samples:,} samples")
```
`num_samples` was defined as 2^19 (for 10 Hz) and 2^17 (for the rest) because of the sampling rate (the number of samples captured per second) used in the picoscope for capturing the data. It's good practise to have a higher sampling rate else we fall victim to aliasing. Aliasing bad. However, there's a sweet spot (I'm not sure what it is for this file data, but > 2^17 did a good job). Lower sampling rate = aliasing & data loss. Higher sampling rate = high computation power and file size (redundant).

**4. Cycles for Visualization**
```python
# Calculate period for 4 cycles
period = 1.0 / fundamental_freq
time_for_4_cycles = 4.0 * period
```
The 4 cycles are only for visual purposes. The actual files compute for as many cycles as your sample rate permits (for a constant sample rate, your cycles increase with frequency).

**5. Plot Functions**
Every function defined as `plot_*` is a visual function; not an analysis function.

**6. Data Units & Voltage Amplifier Gain**
The scope data is assumed to have 3 columns of data: `[time(ms), Voltage(V), Current(A)]`, and across the code, you will notice that I have converted these units to a scale that is comprehensible: `[s, kV, mA]`, as well as MΩ and nF.

**Voltage amplifier gain = 3x** in the code. This is a hardcoded value -- 1:3000 for my test setup; 1 volt step up on my signal generator is a 3kV step up through the amplifier (at the DUT). I urge you to double check if your power amplifier (or transformer) has the same gain before skipping this change. If not, change this value accordingly.

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

### Step 2b: Energy and Power Analysis

**Script:** [`source codes/A5 Energy and Power Analysis.py`](source%20codes/A5%20Energy%20and%20Power%20Analysis.py)

**Why this matters:**
This script calculates the energy dissipation and power consumption of your measured MOV over time. You'll need these results later (after modeling in ATP) to compare against your ATP model's energy and power outputs - this is how you validate that your model actually behaves like the real device. Don't skip this!

**What it does:**
- Calculates energy dissipation and power over time
- Creates detailed energy/power time series plots
- Generates batch comparison plots across frequencies
- Produces a summary file (keep this handy for ATP model comparison later)

**Heads up:** This takes approximately **30-35 minutes** to run for a typical dataset. Touch grass, socialise with others, grab a coffee - the code will be fine without you.

**Configuration:**

| Line | Parameter | Default Value | Description |
|------|-----------|---------------|-------------|
| **Line 1038** | `file_pattern` | `"A5_1_*.csv"` | Input file pattern |
| **Line 1039** | `output_dir` | `"energy_results_peak_start"` | Output directory |
| Line 1040 | `start_from_first_peak` | `True` | Start analysis from first voltage peak |
| Line 1041 | `skip_initial_samples` | `10` | Skip initial transient samples |

#### Understanding the Configurable Parameters

**1. Peak Detection Sensitivity**
```python
min_peak_height_ratio = 0.3
```
Controls how prominent a voltage peak must be to trigger the analysis start point. Default 0.3 means 30% of max voltage. Increase to 0.5+ for noisy data where false peaks might be detected. Lower values = more sensitive, higher values = only catches obvious peaks.

**2. Analysis Start Method**
```python
start_from_first_peak = True
skip_initial_samples = 10
```
Two options: either start from the first detected voltage peak (`True`) or skip a fixed number of samples (`False` + set `skip_initial_samples`). Peak-based start is usually better as it syncs with the actual waveform, but fixed skip is useful if peak detection is being unreliable.

**3. Voltage Amplifier Gain**
Same as A5 Analysis - hardcoded 3x gain. Change if your setup differs.

**How to run:**
```bash
python "source codes/A5 Energy and Power Analysis.py"
```

**Outputs:**
- Energy and power time series CSV files
- **Batch comparison plots** (3-4 plots) - compare across frequencies
- **Summary CSV with energy metrics** - glance through this, you'll need it when doing ATP model energy/power comparison later

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

#### Understanding the Configurable Parameters

**1. Nonlinear Resistance Threshold**
```python
nonlinear_res_threshold = 900  # MΩ
```
Filters out data points where nonlinear resistance exceeds this value. Why? At very low currents, the calculated nonlinear resistance becomes unrealistically high (approaching infinity) and skews the polynomial fit. 900 MΩ works well for most cases, but if you're losing valid data points, increase it. If you're getting weird polynomial fits, try decreasing it.

**2. Type 92 Table Points**
```python
type92_points = 20
```
Number of V-I points generated for the ATP/EMTP Type 92 element. 20 points is the preferred value. Also good to read the ATP-EMTP Manual and rulebook for this (see section on modelling in the `modelling/` folder - coming soon).

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

#### Understanding the Configurable Parameters

**1. R-Priority Mode**
```python
prioritize_R = True  # or False
```
The script runs twice automatically - once balanced, once R-prioritised. Honestly? I just use R-priority because it makes me happier. But the actual reason it exists: the R-priority mode weights resistance fitting 5× higher than capacitance in the error minimisation. This helps when the optimiser might otherwise sacrifice R accuracy to get a better X fit - and we want those R losses properly captured in the model.

**2. Input Units**
The script expects `R_equ` and `|X_equ|` columns in **MΩ**. It converts internally to Ω. If your data is already in Ω, your results will be off by 10^6 - double check this!

**3. Optimization Parameters (advanced)**
```python
maxiter = 5000   # Maximum iterations
popsize = 50     # Population size
```
Don't touch these unless you know what you're doing. See Appendix of the thesis to learn more about differential evolution parameters and which ones affect what.

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

#### Understanding the Configurable Parameters

**1. Number of RC Branches (N)**
```python
N = 3
```
N=3 was the initial model in the thesis. I also tried N=5 and saw better accuracy in the frequency response match - but obviously the component values start getting absurd. More branches = closer match to the ideal Cole-Cole curve, but how willing are you to play with data-entry is the real question. The function of these RC parameters is explained in Chapter 4 (Modelling) of the thesis.

**2. Frequency Range (f0, f1)**
```python
f0 = 1      # Hz - lower bound
f1 = 1000   # Hz - upper bound
```
Extend by 1-2 decades on each side of your measurement range to avoid approximation errors at corner frequencies.
- Example: If you measured 10-500 Hz, use f0=1 Hz and f1=5000 Hz

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

## Tips (very imp.)

1. **Keep organised:** Create separate folders for each voltage/frequency sweep
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


Questions, feedback, or just want to chat about surge arresters? Feel free to reach out!

- **Email:** [pavanpratyush@gmail.com](mailto:pavanpratyush@gmail.com)
- **LinkedIn:** [Pavan Dhulipala](https://linkedin.com/in/pavanpratyush/)


---

*Last updated: November 2025*
