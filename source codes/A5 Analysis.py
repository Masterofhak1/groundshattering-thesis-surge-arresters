import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from numba import njit
import warnings
import glob
import os
import pandas as pd
import matplotlib.animation as animation
import tempfile
import scipy.io
from scipy.interpolate import pchip_interpolate
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from scipy import stats

# Global variable to store the selected method
# Fixed to use sine wave method for capacitive current calculation
warnings.filterwarnings("ignore", category=np.ComplexWarning)

# ===============================================================
# Helper Functions
# ===============================================================

def standardize_frequency(freq):
    """
    Standardize frequency values to ensure consistent categorization.
    For frequencies near the standard target values, use the exact target value.
    Otherwise, round to the nearest integer.
    """
    target_freqs = [10, 17, 27, 50, 100, 150, 300, 500]
    for target_freq in target_freqs:
        if abs(freq - target_freq) <= 1.5:
            return target_freq
    return round(freq)

@njit
def goertzelAtFreq(x, fTest, Fs):
    """
    Goertzel algorithm to compute the DFT at frequency fTest (JIT compiled).
    Normalized by N for consistency with typical DFT amplitude scaling.
    """
    N = len(x)
    w0 = 2.0 * np.pi * (fTest / Fs)
    c = 2.0 * np.cos(w0)
    s_prev = 0.0 + 0.0j
    s_prev2 = 0.0 + 0.0j
    for n in range(N):
        s = x[n] + c * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    X = s_prev - np.exp(-1j * w0) * s_prev2
    return X / N

def moving_average(x, window_size):
    """Simple moving average filter."""
    if window_size <= 1:
        return x
    kernel = np.ones(window_size) / float(window_size)
    return np.convolve(x, kernel, mode='same')

def preprocess(x, tx, filename=None, fundamental_freq=None):
    Fs = 1.0 / np.mean(np.diff(tx))
    print(f"Sampling frequency: {Fs:.2f} Hz")
    
    # Default window size (0.0001s)
    window_size = int(round(0.0001 * Fs))
  
    if window_size < 1:
        window_size = 1
        
    return moving_average(x, window_size)

# ===============================================================
# Frequency Estimation (3-point Hann)
# ===============================================================

def iDFT3pHann(record, ts):
    """
    3-point Hann-windowed iDFT method for fundamental frequency estimation.
    
    Parameters:
    ----------
    record : array_like
        The sampled input signal
    ts : float
        Total duration in seconds = (time[-1] - time[0])
    
    Returns:
    -------
    f : float
        Estimated fundamental frequency (Hz)
    A : float
        Estimated amplitude
    P : float
        Estimated phase (radians)
    O : float
        Estimated DC offset
    """
    # Remove DC offset
    O = np.mean(record)
    x = record - O
    
    # Number of samples
    N = len(x)
    
    # Hann window
    t = np.linspace(0, 2*np.pi*(N-1)/N, N)
    w = 0.5 - 0.5*np.cos(t)

    # Performa hann window convolution
    conv = x*w
    
    # Apply window and compute FFT
    sp = np.fft.fft(conv)
    Sp = sp[:N//2]
    M = np.abs(Sp)
    
    # Find peak index (skip DC)
    k = np.argmax(M[1:]) + 1
    
    # Ensure not at edge
    if k == 0:
        k = 1
    if k >= len(M)-1:
        k = len(M)-2
    
    # Frequency bin
    f_bin = k / ts
    
    # 3-point interpolation
    denom = (2*M[k] - M[k-1] - M[k+1])
    if abs(denom) < 1e-10:
        denom = 1e-10  # Prevent division by zero
    
    delta = 0.5*(M[k+1] - M[k-1]) / denom
    
    # Refined frequency estimate
    f = (k + delta) / ts
    
    # Check if close to standard frequency
    target_freqs = [10, 17, 27, 50, 100, 150, 300, 500]
    for target_freq in target_freqs:
        if abs(f - target_freq) <= 0.5:
            print(f"Frequency close to {target_freq} Hz standard value: {f:.2f} Hz")
    
    # Amplitude correction for Hann window
    window_correction = 2.0
    A = window_correction * M[k] / (N/2)
    
    # Phase with correction for window effect
    P = np.angle(Sp[k]) - delta*np.pi*(N-1)/N
    P = (P + np.pi) % (2*np.pi) - np.pi  # Wrap to Â±Ï€
    
    print(f"3-point iDFT with Hann window frequency estimate: {f:.2f} Hz")
    print(f"Phase estimate: {P:.2f} rad, Amplitude: {A:.2f}")
    
    return f, A, P, O

# ===============================================================
# Integrated Voltage and Current Decomposition
# ===============================================================

def integrate_voltage(time, voltage):
    """
    Integrate voltage to reduce noise for phase determination.
    Uses trapezoidal integration after removing DC offset.
    
    Parameters:
    ----------
    time : array_like
        Time vector
    voltage : array_like
        Voltage vector
    
    Returns:
    -------
    integrated_voltage : array_like
        Integrated voltage (flux linkage)
    """
    # Remove DC offset
    voltage_ac = voltage - np.mean(voltage)
    
    # Calculate dt for proper scaling
    dt = np.mean(np.diff(time))
    
    # Integrate using cumulative trapezoidal rule
    integrated = cumtrapz(voltage_ac, dx=dt, initial=0.0)
    
    # Normalize to similar magnitude as original for easier visualization
    scale_factor = np.max(np.abs(voltage)) / np.max(np.abs(integrated))
    integrated = integrated * scale_factor
    
    return integrated

def find_zero_crossings(time, signal, direction='rising'):
    """
    Find zero crossing times of a signal.
    
    Parameters:
    ----------
    time : array_like
        Time vector
    signal : array_like
        Signal to find zero crossings
    direction : str
        'rising', 'falling', or 'both'
    
    Returns:
    -------
    crossings : list
        List of time values where signal crosses zero
    """
    crossings = []
    
    for i in range(len(signal) - 1):
        if direction == 'rising' and signal[i] <= 0 and signal[i+1] > 0:
            # Linear interpolation for exact crossing time
            t0, t1 = time[i], time[i+1]
            v0, v1 = signal[i], signal[i+1]
            t_cross = t0 + (0 - v0) * (t1 - t0) / (v1 - v0)
            crossings.append(t_cross)
        elif direction == 'falling' and signal[i] >= 0 and signal[i+1] < 0:
            t0, t1 = time[i], time[i+1]
            v0, v1 = signal[i], signal[i+1]
            t_cross = t0 + (0 - v0) * (t1 - t0) / (v1 - v0)
            crossings.append(t_cross)
        elif direction == 'both' and ((signal[i] <= 0 and signal[i+1] > 0) or 
                                     (signal[i] >= 0 and signal[i+1] < 0)):
            t0, t1 = time[i], time[i+1]
            v0, v1 = signal[i], signal[i+1]
            t_cross = t0 + (0 - v0) * (t1 - t0) / (v1 - v0)
            crossings.append(t_cross)
    
    return crossings

def reconstructCapacitiveCurrentSine(time, voltage, current, fundamental_freq):
    """
    Reconstruct an ideal capacitive current that leads voltage by +90Â°,
    using voltage zero crossings to anchor the phase and amplitude.
    
    Parameters:
    ----------
    time : array_like
        Time vector
    voltage : array_like
        Voltage vector
    current : array_like
        Total current vector (used for amplitude estimation)
    fundamental_freq : float
        Fundamental frequency in Hz
    
    Returns:
    -------
    i_cap : array_like
        Reconstructed capacitive current
    """
    omega = 2*np.pi*fundamental_freq
    
    # Find rising zero crossings of voltage
    zero_cross_times = find_zero_crossings(time, voltage, 'rising')
    
    if not zero_cross_times:
        print("Warning: No voltage zero crossing found. i_cap set to zero.")
        return np.zeros_like(time)
    
    # Refine frequency from average zero-cross spacing if enough crossings
    if len(zero_cross_times) >= 3:
        periods = np.diff(zero_cross_times)
        avg_period = np.mean(periods)
        freq_refined = 1.0 / avg_period
        
        # Check if refined frequency is within reasonable range
        if abs(freq_refined - fundamental_freq) < 0.2*fundamental_freq:
            print(f"Refined frequency from zero-crossings: {freq_refined:.2f} Hz")
            fundamental_freq = freq_refined
            omega = 2*np.pi*fundamental_freq
    
    # Use second zero crossing as reference if available (skip initial transient)
    ref_idx = min(1, len(zero_cross_times)-1)
    t_ref = zero_cross_times[ref_idx]
    
    # At voltage zero crossing (rising), sin(ωt + φ) = 0, so φ = -ωt_ref
    # Capacitive current leads voltage by 90°, so add π/2
    cap_phase = -omega*t_ref + np.pi/2
    
    # Get current values at voltage zero crossings
    i_at_cross = [np.interp(tz, time, current) for tz in zero_cross_times]
    i_cap_peak = np.mean(np.abs(i_at_cross))
    
    print(f"Capacitive current peak from zero-crossings: {i_cap_peak*1000:.2f} mA")
    
    # Create capacitive current
    i_cap = i_cap_peak * np.sin(omega*time + cap_phase)
    
    # Check if phase should be flipped
    # At voltage zero crossing (rising), capacitive current should be at maximum
    sign_list = [np.sign(np.interp(tz, time, current)) for tz in zero_cross_times]
    if np.mean(sign_list) < 0:
        print("Inverting capacitive current phase based on zero-crossing analysis")
        i_cap = -i_cap
    
    # Verification
    cap_at_zeros = [np.interp(tz, time, i_cap) for tz in zero_cross_times]
    print(f"Verification - Cap current at zero-crossings: {np.mean(np.abs(cap_at_zeros))*1000:.2f} mA")
    
    return i_cap

# ===============================================================
# DC Measurement Analysis
# ===============================================================

def process_dc_file(dataFile):
    """
    Process a DC measurement file, returning DC average results.
    Uses mean (not absolute mean) for noise reduction.
    
    Parameters:
    ----------
    dataFile : str
        Path to the data file
    
    Returns:
    -------
    dict
        Dictionary with DC measurement results
    """
    basename = os.path.basename(dataFile)
    print(f"Processing DC file: {basename}")
    try:
        # Try first with faster numpy loading
        try:
            # Just load the first 10,000 points at most - no need for all data in DC files
            data_array = np.loadtxt(dataFile, delimiter=',', skiprows=3, max_rows=10000)
        except Exception as e:
            print(f"Failed with numpy, trying with pandas: {str(e)}")
            
            # Faster pandas loading - only read necessary columns and subset of rows
            try:
                # Read just the header to find where numeric data starts
                header_df = pd.read_csv(dataFile, nrows=10)
                
                # Determine how many rows to skip
                numeric_start = 0
                for i, row in header_df.iterrows():
                    try:
                        float(row.iloc[0])
                        numeric_start = i
                        break
                    except (ValueError, TypeError):
                        continue
                
                # Now load only a subset of the data
                df = pd.read_csv(dataFile, skiprows=numeric_start, nrows=10000)
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna()
                data_array = df.values
            except Exception as inner_e:
                print(f"Pandas optimization failed, trying standard pandas: {str(inner_e)}")
                # Fall back to original pandas method if optimized version fails
                df = pd.read_csv(dataFile)
                
                # Skip header rows
                numeric_start = 0
                for i, row in df.iterrows():
                    try:
                        float(row.iloc[0])
                        numeric_start = i
                        break
                    except (ValueError, TypeError):
                        continue
                
                print(f"First numeric row starts at index {numeric_start}")
                numeric_df = df.iloc[numeric_start:].reset_index(drop=True)
                for col in numeric_df.columns:
                    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                numeric_df = numeric_df.dropna()
                data_array = numeric_df.values
        
        if data_array.shape[0] < 100:
            raise ValueError(f"Not enough data points in DC file: {data_array.shape[0]} < 100")
        
        # Get data
        time_all = data_array[:, 0] * 0.001  # ms to s
        ch1_all = data_array[:, 1]          # Voltage in kV (CH1)
        ch3_all = data_array[:, 2]          # Current in mA (CH3)
        
        # Preprocess and convert to SI units
        ch1_all_proc = ch1_all * 1000 * 3  # kV -> V, 3x amplifier
        ch3_all_proc = ch3_all * 1e-3      # mA -> A
        
        # Calculate DC values using mean (not absolute mean) for better noise reduction
        v_dc = np.mean(ch1_all_proc)
        i_dc = np.mean(ch3_all_proc)
        r_dc = v_dc / i_dc if i_dc > 0 else np.nan
        
        # Return DC results
        return {
            'filename': basename,
            'v_dc': v_dc,
            'i_dc': i_dc,
            'r_dc': r_dc,
            'is_dc': True,
            'Voltage (kV)': v_dc / 1000,  # Convert to kV
            'Current (mA)': i_dc * 1000,  # Convert to mA
            'Rs (MΩ)': r_dc / 1e6,        # Convert to MΩ
        }
    
    except Exception as e:
        print(f"Error processing DC file {basename}: {str(e)}")
        raise

# ===============================================================
# Main AC Analysis Function
# ===============================================================

def analyze_mov_file_improved(dataFile):
    """
    Enhanced MOV analysis using integration-based phase shift method.
    1. Reconstruct capacitive current at voltage zero-crossings
    2. Use integration to reduce noise in phase determination
    
    Parameters:
    ----------
    dataFile : str
        Path to the data file
    
    Returns:
    -------
    dict
        Dictionary with analysis results
    """
    basename = os.path.basename(dataFile)
    print(f"Processing {basename}...")
    
    # Checks if this is a DC file
    if "DC" in basename.upper():
        return process_dc_file(dataFile)

    # Sine wave method for capacitive current calculation
    method = "sine"
    print("Using sine wave method for capacitive current calculation")
    
    try:
        # Try loading with numpy first
        try:
            print("Attempting to load with numpy...")
            data_array = np.loadtxt(dataFile, delimiter=',', skiprows=3)
        except Exception as e:
            print(f"Failed with numpy, trying with pandas: {str(e)}")
            # Fallback to pandas
            df = pd.read_csv(dataFile)
            
            # Skip header rows
            numeric_start = 0
            for i, row in df.iterrows():
                try:
                    float(row.iloc[0])
                    numeric_start = i
                    break
                except (ValueError, TypeError):
                    continue
            
            print(f"First numeric row starts at index {numeric_start}")
            
            # Extract only numeric data
            numeric_df = df.iloc[numeric_start:].reset_index(drop=True)
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
            numeric_df = numeric_df.dropna()
            
            data_array = numeric_df.values
        
        if data_array.shape[0] < 10000:
            raise ValueError(f"Not enough data points in file: {data_array.shape[0]} < 10000")
        
         # Match the original indexing style
        time_all = data_array[:, 0] * 0.001  # ms to s
        ch1_all = data_array[:, 1]          # Voltage in kV (CH1)
        ch3_all = data_array[:, 2]          # Current in mA (CH3)
        
        # Preprocess and convert to SI units
        time_all_proc = time_all
        ch1_all_proc = preprocess(ch1_all, time_all, basename) * 1000 * 3  # kV -> V, 3x amplifier
        ch3_all_proc = preprocess(ch3_all, time_all, basename) * 1e-3      # mA -> A
        
        # Remove DC bias
        ch1_debiased = ch1_all_proc - np.mean(ch1_all_proc)
        ch3_debiased = ch3_all_proc - np.mean(ch3_all_proc)
        
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

        end_idx = min(start_idx + num_samples - 1, len(time_all_proc) - 1)
        
        print(f"Extracting segment from index {start_idx} to {end_idx} (total: {end_idx - start_idx + 1} points)")
        
        time = time_all_proc[start_idx:end_idx+1]
        voltage = ch1_debiased[start_idx:end_idx+1]
        current = ch3_debiased[start_idx:end_idx+1]

        # Sanity Check to check how many samples are being extracted
        print(f"EXTRACTION CHECK: Got {len(time)} samples, {time[-1]-time[0]:.1f} seconds")

        # Calculate sampling frequency
        Fs = 1 / np.mean(np.diff(time))
        
        # Use 3-point Hann windowed iDFT for improved frequency estimation
        fundamental_freq, voltage_amplitude, voltage_phase, _ = iDFT3pHann(voltage, time[-1] - time[0])
        
        # Store the standardized frequency
        standardized_freq = standardize_frequency(fundamental_freq)
        
        # Generate integrated voltage for phase reference
        voltage_int = integrate_voltage(time, voltage)
        
        # Step 1: Reconstruct capacitive current using voltage zero-crossings
        # Step 1: Calculate capacitance using sine method first (needed for frequency method)
        i_cap_sine = reconstructCapacitiveCurrentSine(time, voltage, current, fundamental_freq)

        # Calculate capacitance from sine-based method
        omega = 2 * np.pi * fundamental_freq
        v_peak = np.max(np.abs(voltage))
        i_cap_sine_peak = np.max(np.abs(i_cap_sine))
        Cs = i_cap_sine_peak / (v_peak * omega) if (v_peak * omega) > 0 else np.nan

        print(f"Estimated capacitance: {Cs*1e9:.2f} nF")

        # Use the selected method for capacitive current calculation
        # Always use sine wave method
        i_cap = i_cap_sine
        
        # Step 2: Calculate remainder current (total - capacitive)
        i_res = current - i_cap

        # Step 3: Apply simple smoothing to resistive current
        i_res = smooth_resistive_current(i_res, time)
        
        # Calculate peak values
        v_peak = np.max(np.abs(voltage))
        i_peak = np.max(np.abs(current))
        i_cap_peak = np.max(np.abs(i_cap))
        i_res_peak = np.max(np.abs(i_res))
        
        # Calculate resistance and capacitance
        Rs = v_peak / i_res_peak if i_res_peak > 0 else np.nan
        omega = 2 * np.pi * fundamental_freq
        Cs = i_cap_peak / (v_peak * omega) if (v_peak * omega) > 0 else np.nan
        
        # Store waveform data for plotting
        waveform_data = {
            'time': time,
            'voltage': voltage,
            'voltage_int': voltage_int,
            'current': current,
            'i_cap': i_cap,
            'i_res': i_res,
            'v_peak': v_peak,
            'i_peak': i_peak,
            'i_cap_peak': i_cap_peak,
            'i_res_peak': i_res_peak,
            'Rs': Rs,
            'Cs': Cs,
            'fundamental_freq': fundamental_freq,
            'standardized_freq': standardized_freq,
            'method_used': 'sine',  # To choose method
            'is_linear_ref': False,  # Will be updated during decomposition
            'i_linear': None,        # Will be updated during decomposition
            'i_nonlinear': None      # Will be updated during decomposition
        }
        
        # Package results for return
        result = {
            'filename': basename,
            'freq': fundamental_freq,
            'standardized_freq': standardized_freq,
            'v_peak': v_peak,
            'i_peak': i_peak,
            'i_cap_peak': i_cap_peak,
            'i_res_peak': i_res_peak,
            'Rs': Rs,
            'Cs': Cs,
            'method_used': 'sine',  
            'cap_percent': (i_cap_peak / i_peak) * 100,
            'res_percent': (i_res_peak / i_peak) * 100,
            'waveform_data': waveform_data,
            'is_dc': False
        }
        
        # Print summary
        print('\n=== MOV ANALYSIS RESULTS ===')
        print(f'File: {basename}')
        print(f'Peak Voltage: {v_peak:.2f} V ({v_peak/1000:.2f} kV)')
        print(f'Peak Current: {i_peak*1000:.2f} mA')
        print(f'Fundamental Frequency: {fundamental_freq:.2f} Hz (Standardized: {standardized_freq:.2f} Hz)')
        print(f'Peak Capacitive Current: {i_cap_peak*1000:.2f} mA ({(i_cap_peak/i_peak)*100:.2f}%)')
        print(f'Peak Resistive Current: {i_res_peak*1000:.2f} mA ({(i_res_peak/i_peak)*100:.2f}%)')
        print(f'Capacitance: {Cs*1e9:.2f} nF')
        print(f'Resistance: {Rs/1e6:.2f} MΩ')
        
        return result
    
    except Exception as e:
        print(f'Error processing {basename}: {str(e)}')
        raise

# ===============================================================
# Data Processing
# ===============================================================

def smooth_resistive_current(i_res, time):
    """
    Apply a constant 0.0001s window smoothing to resistive current.
    
    Parameters:
    ----------
    i_res : array_like
        Resistive current to smooth
    time : array_like
        Time vector
    
    Returns:
    -------
    array_like
        Smoothed resistive current
    """
    # Calculate sampling frequency
    Fs = 1.0 / np.mean(np.diff(time))
    
    # Use a constant 0.0001s window
    window_size = int(round(0.0001 * Fs))
    
    # Ensure minimum window size
    if window_size < 3:
        window_size = 3
    
    print(f"Smoothing resistive current with {window_size} samples window ({window_size/Fs*1000:.3f} ms)")
    
    # Apply smoothing
    smoothed = moving_average(i_res, window_size)
    
    return smoothed

def process_all_files(file_pattern):
    """
    Find and process all files matching the pattern.
    
    Parameters:
    ----------
    file_pattern : str
        Glob pattern to match files
    
    Returns:
    -------
    tuple
        (df_ac, df_dc) - DataFrames with AC and DC results
    """
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        print(f'No files found matching pattern: {file_pattern}')
        return None, None
    
    print(f"Found {len(file_list)} files to process.")
    
    all_results = []
    for filename in file_list:
        try:
            result = analyze_mov_file_improved(filename)
            all_results.append(result)
            print(f'Processed {os.path.basename(filename)} successfully.')
        except Exception as e:
            print(f'Error processing {os.path.basename(filename)}: {str(e)}')
    
    if all_results:
        # Separate AC and DC results
        ac_results = [r for r in all_results if not r.get('is_dc', False)]
        dc_results = [r for r in all_results if r.get('is_dc', False)]
        
        # Create DataFrames
        df_ac = pd.DataFrame(ac_results) if ac_results else None
        df_dc = pd.DataFrame(dc_results) if dc_results else None
        
        return df_ac, df_dc
    else:
        return None, None

def align_waveforms_with_correlation(reference_wave, target_wave):
    """
    Aligns two waveforms using cross-correlation to find the optimal phase shift.
    More robust than simple zero-crossing detection.
    
    Parameters:
    ----------
    reference_wave : array_like
        The reference waveform to be shifted
    target_wave : array_like
        The target waveform to align with
    
    Returns:
    -------
    array_like
        Phase-aligned reference waveform
    float
        Correlation coefficient after alignment
    """
    import numpy as np
    from scipy import signal
    
    # Ensure both waveforms have the same length
    assert len(reference_wave) == len(target_wave), "Waveforms must have the same length"
    
    # Normalize waveforms to zero mean and unit variance for better correlation
    ref_norm = (reference_wave - np.mean(reference_wave)) / (np.std(reference_wave) + 1e-10)
    target_norm = (target_wave - np.mean(target_wave)) / (np.std(target_wave) + 1e-10)
    
    # Compute cross-correlation
    correlation = signal.correlate(target_norm, ref_norm, mode='full')
    
    # Find the lag with maximum correlation
    max_corr_idx = np.argmax(correlation)
    lag = max_corr_idx - (len(ref_norm) - 1)
    
    # Shift the reference waveform by the lag
    if lag > 0:
        aligned_ref = np.concatenate((reference_wave[lag:], reference_wave[:lag]))
    else:
        aligned_ref = np.concatenate((reference_wave[lag:], reference_wave[:lag]))
    
    # Calculate correlation coefficient after alignment
    corr_coef = np.corrcoef(aligned_ref, target_wave)[0, 1]
    
    return aligned_ref, corr_coef

def extract_linear_nonlinear_resistance(df_ac):
    """
    Improved function to extract linear and non-linear resistance components
    using cross-correlation for better phase alignment.
    """
    if df_ac is None or df_ac.empty:
        print("No AC data to process for resistance decomposition.")
        return df_ac
    
    df = df_ac.copy()
    
    df['R_linear'] = 0.0
    df['R_nonlinear'] = 0.0
    df['R_linear (MΩ)'] = 0.0
    df['R_nonlinear (MΩ)'] = 0.0
    df['alignment_quality'] = 0.0  # Track alignment quality for debugging
    
    unique_freqs = df['Frequency (Hz)'].unique()
    
    for freq in unique_freqs:
        freq_group = df[df['Frequency (Hz)'] == freq].copy()
        freq_group = freq_group.sort_values('Voltage (kV)')
        
        if len(freq_group) == 0:
            continue
        
        # Get lowest voltage data as reference
        ref_row = freq_group.iloc[0]
        ref_file = ref_row['filename']
        R_linear = ref_row['Rs']
        
        waveform_data = ref_row['waveform_data']
        waveform_data['is_linear_ref'] = True
        
        # Get reference data
        ref_time = waveform_data['time']
        ref_voltage = waveform_data['voltage']
        ref_i_res = waveform_data['i_res']
        
        print(f"\nFrequency {freq} Hz - Reference file: {ref_file}")
        print(f"  Reference voltage: {ref_row['Voltage (kV)']:.3f} kV")
        
        # Process each file in this frequency group
        for idx, row in freq_group.iterrows():
            row_waveform = row['waveform_data']
            time = row_waveform['time']
            voltage = row_waveform['voltage']
            i_res = row_waveform['i_res']
            v_peak = row_waveform['v_peak']
            
            # Scale amplitude based on voltage ratio
            voltage_ratio = v_peak / ref_row['v_peak']
            
            # IMPROVEMENT: Cross-correlation based alignment
            if len(ref_i_res) == len(i_res):
                # Use cross-correlation to find the optimal phase shift
                amplitude_scaled_ref = ref_i_res * voltage_ratio
                i_linear, corr_coef = align_waveforms_with_correlation(amplitude_scaled_ref, i_res)
                
                # Print alignment quality for debugging
                print(f"  {row['filename']} - Alignment correlation: {corr_coef:.3f}")
                df.loc[idx, 'alignment_quality'] = corr_coef
                
                # If correlation is very poor (less than 0.5), try zero-crossing based alignment
                if corr_coef < 0.5 and len(i_res) > 100:
                    print(f"    Low correlation detected! Trying zero-crossing alignment.")
                    
                    # Find zero crossings of both signals
                    zeros_ref = np.where(np.diff(np.signbit(ref_i_res)))[0]
                    zeros_target = np.where(np.diff(np.signbit(i_res)))[0]
                    
                    if len(zeros_ref) > 0 and len(zeros_target) > 0:
                        # Use first zero crossing in each signal to calculate phase shift
                        shift = zeros_target[0] - zeros_ref[0]
                        
                        # Shift the reference waveform
                        i_linear = np.roll(amplitude_scaled_ref, shift)
                        
                        # Check the correlation after zero-crossing alignment
                        new_corr = np.corrcoef(i_linear, i_res)[0, 1]
                        print(f"    Zero-crossing alignment correlation: {new_corr:.3f}")
                        
                        # Only use if it's better
                        if new_corr > corr_coef:
                            corr_coef = new_corr
                            df.loc[idx, 'alignment_quality'] = corr_coef
                        else:
                            # Revert to cross-correlation result if not better
                            i_linear, _ = align_waveforms_with_correlation(amplitude_scaled_ref, i_res)
            else:
                # For different length waveforms, use time normalization and interpolation
                print(f"  {row['filename']} - Different length waveforms, using interpolation")
                ref_period = 1 / waveform_data['fundamental_freq']
                this_period = 1 / row_waveform['fundamental_freq']
                
                # Create normalized time vectors (0 to 1 cycle)
                ref_t_norm = (ref_time - ref_time[0]) % ref_period / ref_period
                t_norm = (time - time[0]) % this_period / this_period
                
                # Calculate optimal phase shift using cross-correlation on downsampled signals
                ref_resampled = np.interp(np.linspace(0, 1, 1000), ref_t_norm, ref_i_res) * voltage_ratio
                target_resampled = np.interp(np.linspace(0, 1, 1000), t_norm, i_res)
                
                # Find optimal phase shift using correlation
                correlation = np.correlate(target_resampled, ref_resampled, mode='full')
                phase_shift = (np.argmax(correlation) - (len(ref_resampled) - 1)) / len(ref_resampled)
                
                # Apply phase shift and interpolate
                shifted_t_norm = (t_norm + phase_shift) % 1.0
                i_linear = np.interp(t_norm, shifted_t_norm, ref_i_res) * voltage_ratio
                
                # Calculate correlation coefficient
                corr_coef = np.corrcoef(i_linear, i_res)[0, 1]
                df.loc[idx, 'alignment_quality'] = corr_coef
                print(f"  Interpolation alignment correlation: {corr_coef:.3f}")
            
            # Calculate nonlinear component
            i_nonlinear = i_res - i_linear
            
            # Store in waveform data
            row_waveform['i_linear'] = i_linear
            row_waveform['i_nonlinear'] = i_nonlinear
            row_waveform['i_linear_peak'] = np.max(np.abs(i_linear))
            row_waveform['i_nonlinear_peak'] = np.max(np.abs(i_nonlinear))
            
            # Update percentage values for title
            i_res_peak = np.max(np.abs(i_res))
            i_linear_peak = np.max(np.abs(i_linear))
            i_nonlinear_peak = np.max(np.abs(i_nonlinear))
            
            linear_percent = (i_linear_peak / i_res_peak) * 100 if i_res_peak > 0 else 0
            nonlinear_percent = (i_nonlinear_peak / i_res_peak) * 100 if i_res_peak > 0 else 0
            
            row_waveform['linear_percent'] = linear_percent
            row_waveform['nonlinear_percent'] = nonlinear_percent
            
            # Linear resistance remains constant across voltages
            R_linear_actual = R_linear
            
            # Calculate non-linear component resistance
            R_total = row['Rs']
            if R_total < R_linear_actual:
                R_nonlinear = (R_linear_actual * R_total) / (R_linear_actual - R_total)
            else:
                R_nonlinear = 1e12  # Cap very high resistances
            
            # Update DataFrame
            df.loc[idx, 'R_linear'] = R_linear_actual
            df.loc[idx, 'R_nonlinear'] = R_nonlinear
            df.loc[idx, 'R_linear (MΩ)'] = R_linear_actual / 1e6
            df.loc[idx, 'R_nonlinear (MΩ)'] = R_nonlinear / 1e6
    
    # Check for any poor alignments and warn
    poor_alignments = df[df['alignment_quality'] < 0.7]
    if not poor_alignments.empty:
        print("\nWARNING: Poor waveform alignments detected in these files:")
        for _, row in poor_alignments.iterrows():
            print(f"  {row['filename']} - Correlation: {row['alignment_quality']:.3f}")
        print("Consider inspecting these files manually.")
    
    return df

# ===============================================================
# Visualization Functions
# ===============================================================

def plotResults(waveform_data, basename, output_dir):
    """
    Enhanced plotResults function that also saves raw waveform data to CSV.
    Uses standardized frequency for folder creation.
    """
    # Unpack waveform data
    time = waveform_data['time']
    voltage = waveform_data['voltage']
    voltage_int = waveform_data.get('voltage_int', None)
    current = waveform_data['current']
    i_cap = waveform_data['i_cap']
    i_res = waveform_data['i_res']
    Rs = waveform_data['Rs']
    Cs = waveform_data['Cs']
    v_peak = waveform_data['v_peak']
    i_peak = waveform_data['i_peak']
    fundamental_freq = waveform_data['fundamental_freq']
    
    # Use standardized frequency for directory name
    standardized_freq = waveform_data.get('standardized_freq', standardize_frequency(fundamental_freq))
    
    i_linear = waveform_data.get('i_linear', None)
    i_nonlinear = waveform_data.get('i_nonlinear', None)
    
    # Skip first 20% of the data to avoid initial transients?
    start_offset = int(len(time) * 0.2)
    
    # Calculate period for 4 cycles
    period = 1.0 / fundamental_freq
    time_for_4_cycles = 4.0 * period
    
    # Find the first positive zero-crossing of voltage after the offset
    for i in range(start_offset, len(voltage) - 1):
        if voltage[i] <= 0 and voltage[i+1] > 0:
            plot_start_idx = i
            break
    else:
        plot_start_idx = start_offset
    
    plot_start_time = time[plot_start_idx]
    plot_end_time = plot_start_time + time_for_4_cycles
    
    plot_end_idx = plot_start_idx
    for i in range(plot_start_idx, len(time)):
        if time[i] >= plot_end_time:
            plot_end_idx = i
            break
    else:
        plot_end_idx = len(time) - 1
    
    plot_time = time[plot_start_idx:plot_end_idx+1]
    plot_voltage = voltage[plot_start_idx:plot_end_idx+1]
    plot_voltage_int = voltage_int[plot_start_idx:plot_end_idx+1] if voltage_int is not None else None
    plot_current = current[plot_start_idx:plot_end_idx+1]
    plot_i_cap = i_cap[plot_start_idx:plot_end_idx+1]
    plot_i_res = i_res[plot_start_idx:plot_end_idx+1]
    plot_i_linear = i_linear[plot_start_idx:plot_end_idx+1] if i_linear is not None else None
    plot_i_nonlinear = i_nonlinear[plot_start_idx:plot_end_idx+1] if i_nonlinear is not None else None
    
    # Create output directory with safe frequency name (no decimals)
    freq_dir = f"{output_dir}/{standardized_freq:.1f}Hz".replace('.', 'p')
    os.makedirs(freq_dir, exist_ok=True)
    
    # Plotting and saving to PNG
    try:
        # Normalize for plotting
        v_norm = plot_voltage / np.max(np.abs(plot_voltage))
        
        # Create figure with 3 subplots
        plt.figure(figsize=(12, 12))
        
        # First subplot (normalized waveforms)
        plt.subplot(3, 1, 1)
        plt.plot(plot_time, v_norm, 'k', linewidth=1.0, label='Voltage (norm)')
        plt.plot(plot_time, plot_current / np.max(np.abs(plot_current)),
                'b', linewidth=1.0, label='Total Current (norm)')
        plt.plot(plot_time, plot_i_cap / np.max(np.abs(plot_i_cap)),
                'g', linewidth=1.0, label='Cap. Current (norm)')

        plt.legend(loc='upper right')
        plt.title(f'Voltage & Current Waveforms (Normalised) - 4 Cycles\n'
        f'V_peak = {v_peak/1e3:.2f} kV, '
        f'I_peak = {i_peak*1e3:.2f} mA, '
        f'f = {standardized_freq:.1f} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Second subplot (capacitive/resistive decomposition)
        plt.subplot(3, 1, 2)
        plt.plot(plot_time, plot_current * 1e3, 'b', linewidth=1.0, label='Total Current')
        plt.plot(plot_time, plot_i_cap * 1e3, 'g', linewidth=1.0, label='Cap. Current')
        plt.plot(plot_time, plot_i_res * 1e3, 'r', linewidth=1.0, label='Res. Current')
        plt.legend(loc='upper right')
        plt.title(f'Current Component Decomposition - 4 Cycles '
                f'(C = {Cs*1e9:.2f} nF, R = {Rs/1e6:.2f} MΩ)')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (mA)')
        plt.grid(True, alpha=0.3)
        
        # Third subplot (linear/nonlinear resistive decomposition)
        plt.subplot(3, 1, 3)
        plt.plot(plot_time, plot_i_res * 1e3, 'r', linewidth=1.0, label='Total Res. Current')
        
        if plot_i_linear is not None and plot_i_nonlinear is not None:
            plt.plot(plot_time, plot_i_linear * 1e3, 'c', linewidth=1.0, label='Linear Res. Current')
            plt.plot(plot_time, plot_i_nonlinear * 1e3, 'm', linewidth=1.0, label='Non-linear Res. Current')
            
            # Calculate percentage of linear and nonlinear components
            i_linear_peak = np.max(np.abs(plot_i_linear))
            i_nonlinear_peak = np.max(np.abs(plot_i_nonlinear))
            i_res_peak = np.max(np.abs(plot_i_res))
            
            linear_percent = (i_linear_peak / i_res_peak) * 100
            nonlinear_percent = (i_nonlinear_peak / i_res_peak) * 100
            
            plt.title(f'Resistive Current Decomposition - 4 Cycles\n'
                    f'Linear: {linear_percent:.1f}%, Non-linear: {nonlinear_percent:.1f}%')
        else:
            plt.title('Resistive Current (Linear/Non-linear decomposition not available)')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Current (mA)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure with the full original filename
        plt.savefig(f"{freq_dir}/{basename}_waveforms.png", dpi=300)
        print(f"Saved waveform plot to {freq_dir}/{basename}_waveforms.png")
        plt.close()
        
        # Create data dictionary for CSV
        data_dict = {
            'time': plot_time,
            'voltage': plot_voltage,
            'current_total': plot_current,
            'current_capacitive': plot_i_cap,
            'current_resistive': plot_i_res
        }
        
        # Add integrated voltage if available
        if plot_voltage_int is not None:
            data_dict['voltage_integrated'] = plot_voltage_int
        
        # Add linear/nonlinear components if available
        if plot_i_linear is not None:
            data_dict['current_linear'] = plot_i_linear
        if plot_i_nonlinear is not None:
            data_dict['current_nonlinear'] = plot_i_nonlinear
        
        # Create the DataFrame
        waveform_df = pd.DataFrame(data_dict)
        
        # Add metadata as additional columns
        waveform_df['v_peak'] = v_peak
        waveform_df['Rs'] = Rs
        waveform_df['Cs'] = Cs
        waveform_df['frequency'] = fundamental_freq
        waveform_df['standardized_frequency'] = standardized_freq
        waveform_df['i_peak'] = i_peak
        waveform_df['i_res_peak'] = np.max(np.abs(plot_i_res))
        waveform_df['i_cap_peak'] = np.max(np.abs(plot_i_cap))
        
        if plot_i_linear is not None and plot_i_nonlinear is not None:
            waveform_df['i_linear_peak'] = np.max(np.abs(plot_i_linear))
            waveform_df['i_nonlinear_peak'] = np.max(np.abs(plot_i_nonlinear))
            waveform_df['linear_percent'] = (waveform_df['i_linear_peak'] / waveform_df['i_res_peak']) * 100
            waveform_df['nonlinear_percent'] = (waveform_df['i_nonlinear_peak'] / waveform_df['i_res_peak']) * 100
        
        # Save to CSV
        waveform_df.to_csv(f"{freq_dir}/{basename}_waveforms.csv", index=False)
        print(f"Saved waveform data to {freq_dir}/{basename}_waveforms.csv")
        
    except Exception as e:
        print(f"Error processing {basename}: {str(e)}")
        plt.close()  # Close the figure if there's an error

def plot_all_waveforms(df, output_dir="results_improved"):
    """Create waveform plots for all processed files."""
    if df is None or df.empty:
        print("No data to plot waveforms.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating waveform plots in {output_dir}...")
    
    for idx, row in df.iterrows():
        try:
            waveform_data = row['waveform_data']
            basename = row['filename']
            
            # Add status to filename if this is a reference file
            ref_status = " (REF)" if waveform_data.get('is_linear_ref', False) else ""
            plot_basename = os.path.splitext(basename)[0] + ref_status
            
            # Call plotResults function to create the actual plots and save CSV
            plotResults(waveform_data, plot_basename, output_dir)
        except Exception as e:
            print(f"Error plotting waveforms for {row['filename']}: {str(e)}")
    
    print("All waveform plots generated.")

def plot_linear_resistance(df, output_dir="results_improved", use_log_scale=False):
    """Plot only the linear resistance component."""
    if df is None or df.empty:
        print("No data to plot linear resistance.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    
    unique_freqs = sorted(df['Frequency (Hz)'].unique())
    cmap = plt.cm.tab10
    
    for i, freq in enumerate(unique_freqs):
        color = cmap(i % 10)
        subset = df[df['Frequency (Hz)'] == freq]
        
        # Sort by voltage
        subset = subset.sort_values('Voltage (kV)')
        
        plt.plot(subset['Voltage (kV)'], subset['R_linear (MΩ)'], 
                'o-', color=color, linewidth=1.5, markersize=5,
                label=f'{freq} Hz')
    
    if use_log_scale:
        plt.yscale('log')
        scale_text = "Log Scale"
    else:
        scale_text = "Linear Scale"
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    plt.grid(True, alpha=0.3)
    plt.xlabel('Voltage (kV)', fontsize=12)
    plt.ylabel('Linear Resistance (MΩ)', fontsize=12)
    #plt.title(f'Linear Resistance Component vs Voltage ({scale_text})', fontsize=14)
    
    plt.legend(loc='upper right', fontsize=12)
    
    filename = f'linear_resistance_vs_voltage_{"log" if use_log_scale else "linear"}.png'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300)
    plt.close()

def plot_nonlinear_resistance_with_trend(df, output_dir="results_improved", use_log_scale=False, cap_threshold=900):
    """
    Plot non-linear resistance with color-matched trend lines that seamlessly
    connect to the measurable values.
    
    Parameters:
    ----------
    df : DataFrame
        DataFrame with results
    output_dir : str
        Output directory
    use_log_scale : bool
        Whether to use log scale for y-axis
    cap_threshold : float
        Threshold to consider a value as "capped" (slightly below 1000 MΩ)
    """
    if df is None or df.empty:
        print("No data to plot non-linear resistance.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    unique_freqs = sorted(df['Frequency (Hz)'].unique())
    cmap = plt.cm.tab10
    
    for i, freq in enumerate(unique_freqs):
        color = cmap(i % 10)
        subset = df[df['Frequency (Hz)'] == freq].copy()
        
        # Sort by voltage
        subset = subset.sort_values('Voltage (kV)')
        
        # Find the points where values transition from capped to measurable
        mask_capped = subset['R_nonlinear (MΩ)'] >= cap_threshold
        mask_measurable = ~mask_capped
        
        # Plot the measurable points with solid line
        if sum(mask_measurable) > 0:
            measurable_voltages = subset.loc[mask_measurable, 'Voltage (kV)'].values
            measurable_resistances = subset.loc[mask_measurable, 'R_nonlinear (MΩ)'].values
            
            plt.plot(
                measurable_voltages, 
                measurable_resistances, 
                'o-', 
                color=color, 
                linewidth=1.5, 
                markersize=5,
                label=f'{freq} Hz'
            )
        
        # Plot the trend line for capped points - using same color as measurable points
        if sum(mask_capped) > 0:
            # Get the capped voltages and resistances
            capped_voltages = subset.loc[mask_capped, 'Voltage (kV)'].values
            
            # Find the transition between capped and measurable (if any)
            if sum(mask_measurable) > 0:
                # Get the first measurable point (by voltage)
                first_measurable_idx = subset[mask_measurable]['Voltage (kV)'].idxmin()
                first_measurable_voltage = subset.loc[first_measurable_idx, 'Voltage (kV)']
                first_measurable_resistance = subset.loc[first_measurable_idx, 'R_nonlinear (MΩ)']
                
                # Create extended voltages array that includes transition points
                # Start with all capped voltages
                extended_voltages = list(capped_voltages)
                
                # Add the first measurable point to create a seamless transition
                extended_voltages.append(first_measurable_voltage)
                
                # Sort voltages to ensure proper line drawing
                extended_voltages.sort()
                
                # Create y-values that trend from high to the first measurable point
                extended_resistances = []
                max_resistance = cap_threshold * 9  # Max value to show on plot
                
                for v in extended_voltages:
                    if v == first_measurable_voltage:
                        # This is the transition point - use the actual value
                        extended_resistances.append(first_measurable_resistance)
                    else:
                        # This is a capped point - calculate a trend value
                        # Distance ratio from transition voltage (0 to 1)
                        v_min = min(capped_voltages)
                        v_max = max(capped_voltages)
                        v_range = max(v_max - v_min, 0.001)  # Avoid division by zero
                        
                        # Scale resistance based on position within voltage range
                        if v < first_measurable_voltage:
                            # For voltages below transition, use exponential decay toward transition
                            ratio = (first_measurable_voltage - v) / v_range
                            resistance = first_measurable_resistance + (max_resistance - first_measurable_resistance) * (np.exp(2 * ratio) - 1) / (np.exp(2) - 1)
                            extended_resistances.append(resistance)
                        else:
                            # Should not happen but just in case
                            extended_resistances.append(first_measurable_resistance)
            else:
                # All points are capped, create a high flat line
                extended_voltages = capped_voltages
                extended_resistances = [cap_threshold * 5] * len(capped_voltages)
            
            # Sort the extended points for proper line drawing
            sorted_pairs = sorted(zip(extended_voltages, extended_resistances))
            extended_voltages = [pair[0] for pair in sorted_pairs]
            extended_resistances = [pair[1] for pair in sorted_pairs]
            
            # Draw the trend line with matching color but dashed line
            plt.plot(
                extended_voltages, 
                extended_resistances, 
                '-.', 
                color=color,  # Same color as the measurable points
                linewidth=2.5,
                alpha=0.8,
                dashes=(3, 3)  # Longer dashes
            )
    
    # Set axis limits
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    
    extended_y_max = cap_threshold * 9  # Match the max value used for the trend line
    
    if use_log_scale:
        plt.yscale('log')
        scale_text = "Log Scale"
        # For log scale, show from 10% of smallest value to extended max
        min_val = df.loc[df['R_nonlinear (MΩ)'] < cap_threshold, 'R_nonlinear (MΩ)'].min() * 0.1
        if min_val <= 0:
            min_val = 0.1  # Safety for log scale
        plt.ylim(min_val, extended_y_max)
    else:
        scale_text = "Linear Scale"
        # For linear scale, show from 0 to extended max
        plt.ylim(0, extended_y_max)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Voltage (kV)', fontsize=12)
    plt.ylabel('Non-linear Resistance (MΩ)', fontsize=12)
    # plt.title(f'Non-linear Resistance Component vs Voltage ({scale_text})', fontsize=14)
    
    # Add annotation to explain the dashed line
    plt.annotate('Dashed lines: Trend of values approaching infinity', 
                xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top', fontsize=10)
    
    plt.legend(loc='upper right', fontsize=12)
    
    filename = f'nonlinear_resistance_vs_voltage_{"log" if use_log_scale else "linear"}_seamless.png'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300)
    plt.close()
    
    print(f"Nonlinear resistance plot with seamless trend lines saved to {output_dir}/{filename}")

def plot_dc_resistance(df_dc, output_dir="results_improved", use_log_scale=False):
    """
    Plot DC resistance vs voltage with transition at maximum resistance point.
    
    Parameters:
    ----------
    df_dc : DataFrame
        DataFrame with DC results
    output_dir : str
        Output directory
    use_log_scale : bool
        Whether to use log scale for y-axis
    """
    if df_dc is None or df_dc.empty:
        print("No DC data to plot resistance.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Sort by voltage
    df_sorted = df_dc.sort_values('Voltage (kV)')
    
    # Find the row with the maximum resistance - this will be our transition point
    max_resistance_idx = df_sorted['Rs (MΩ)'].idxmax()
    max_resistance_row = df_sorted.loc[max_resistance_idx]
    max_resistance = max_resistance_row['Rs (MΩ)']
    transition_voltage = max_resistance_row['Voltage (kV)']
    
    print(f"Maximum DC resistance: {max_resistance:.2f} MΩ at {transition_voltage:.2f} kV")
    
    # Only use data points with voltage >= transition voltage
    reliable_dc = df_sorted[df_sorted['Voltage (kV)'] >= transition_voltage].copy()
    unreliable_dc = df_sorted[df_sorted['Voltage (kV)'] < transition_voltage].copy()
    
    print(f"Using {len(reliable_dc)} points with voltage >= transition voltage")
    print(f"Excluding {len(unreliable_dc)} points with voltage < transition voltage")
    
    # Plot reliable DC resistance points with solid line
    plt.plot(
        reliable_dc['Voltage (kV)'], 
        reliable_dc['Rs (MΩ)'], 
        'o-', 
        color='k', 
        linewidth=1.5, 
        markersize=6,
        label='DC Resistance'
    )
    
    # Create dashed trend line extending to lower voltages
    if not unreliable_dc.empty:
        # Get minimum voltage from unreliable points
        min_voltage = unreliable_dc['Voltage (kV)'].min()
        
        # Create voltage points for trend line
        trend_voltages = np.linspace(min_voltage, transition_voltage, 50)
        
        # Create resistance values that trend upward - similar to how we did it for V-I plot
        # But here we directly create the resistance trend rather than computing it from V/I
        # Resistance should increase gently as voltage decreases
        trend_resistances = max_resistance * (1 + 0.1 * (transition_voltage - trend_voltages) / transition_voltage)
        
        # Plot the trend line as dashed
        plt.plot(
            trend_voltages,
            trend_resistances,
            '--',
            color='k',
            linewidth=1.5,
            alpha=0.8
        )
    
    if use_log_scale:
        plt.yscale('log')
        scale_text = "Log Scale"
    else:
        scale_text = "Linear Scale"
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    plt.grid(True, alpha=0.3)
    plt.xlabel('Voltage (kV)', fontsize=12)
    plt.ylabel('Resistance (MΩ)', fontsize=12)
    plt.title(f'DC Resistance vs Voltage ({scale_text})', fontsize=14)
    
    plt.legend(loc='upper right', fontsize=12)
    
    filename = f'dc_resistance_vs_voltage_{"log" if use_log_scale else "linear"}_improved.png'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300)
    plt.close()
    
    print(f"Improved DC resistance plot saved to {output_dir}/{filename}")

def plot_combined_resistance(df_ac, df_dc, output_dir="results_improved", use_log_scale=False):
    """
    Plot AC and DC resistance vs voltage on the same plot for comparison.
    """
    if (df_ac is None or df_ac.empty) and (df_dc is None or df_dc.empty):
        print("No data to plot combined resistance.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    
    # Set up color palette
    cmap = plt.cm.tab10
    
    # Plot AC data if available
    if df_ac is not None and not df_ac.empty:
        unique_freqs = sorted(df_ac['Frequency (Hz)'].unique())
        
        for i, freq in enumerate(unique_freqs):
            color = cmap(i % 10)
            subset = df_ac[df_ac['Frequency (Hz)'] == freq]
            
            # Sort by voltage
            subset = subset.sort_values('Voltage (kV)')
            
            plt.plot(subset['Voltage (kV)'], subset['Rs (MΩ)'], 
                    'o-', color=color, linewidth=1.5, markersize=5,
                    label=f'{int(freq)} Hz - AC')
    
    # Plot DC data if available
    if df_dc is not None and not df_dc.empty:
        # Sort by voltage
        df_dc_sorted = df_dc.sort_values('Voltage (kV)')
        
        # Use a distinct color for DC data (black)
        plt.plot(df_dc_sorted['Voltage (kV)'], df_dc_sorted['Rs (MΩ)'], 
                'D-', color='k', linewidth=1.8, markersize=6,
                label='DC')
    
    if use_log_scale:
        plt.yscale('log')
        scale_text = "Log Scale"
    else:
        scale_text = "Linear Scale"
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    plt.grid(True, alpha=0.3)
    plt.xlabel('Voltage (kV)', fontsize=12)
    plt.ylabel('Resistance (MΩ)', fontsize=12)
    plt.title(f'AC and DC Resistance vs Voltage ({scale_text})', fontsize=14)
    
    plt.legend(loc='upper right', fontsize=12)
    
    filename = f'combined_resistance_vs_voltage_{"log" if use_log_scale else "linear"}.png'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300)
    plt.close()
    
    print(f"Combined resistance plot saved to {output_dir}/{filename}")

def plot_capacitance_resistance(df, output_dir="results_improved"):
    """
    Create plots for capacitance and resistance in linear and log scales.
    Modified for thesis formatting: legends inside plots, no titles.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
        
    # Get unique frequencies
    unique_freqs = sorted(df['Frequency (Hz)'].unique())
    
    # Colour map for consistent colours across plots
    cmap = plt.cm.tab10
    colours = {freq: cmap(i % 10) for i, freq in enumerate(unique_freqs)}
    
    # 1. Capacitance vs. Voltage (Linear scale)
    plt.figure(figsize=(10, 7))
    
    for freq in unique_freqs:
        subset = df[df['Frequency (Hz)'] == freq]
        plt.scatter(
            subset['Voltage (kV)'], subset['Cs (nF)'],
            label=f'{freq} Hz', s=35, alpha=0.8,
            color=colours[freq], edgecolors='none'
        )
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    
    # Determine y limits for better visualization
    y_min = 0
    y_max = df['Cs (nF)'].max() * 1.1
    plt.ylim(y_min, y_max)
    
    plt.grid(True, alpha=0.3, linestyle='-')
    
    plt.xlabel('Peak voltage [kV]', fontsize=12)
    plt.ylabel('Capacitance [nF]', fontsize=12)
    # Removed title: plt.title('Extracted Capacitance Cs (Linear Scale)', fontsize=14)
    
    # Modified legend placement - inside plot
    plt.legend(
        loc='upper right',  # Automatic best location inside plot
        title="Freq. Hz", title_fontsize=12, fontsize=12,
        framealpha=0.9  # Slight transparency to not obscure data
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/capacitance_voltage_linear.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Capacitance vs. Voltage (Log scale)
    plt.figure(figsize=(10, 7))
    
    for freq in unique_freqs:
        subset = df[df['Frequency (Hz)'] == freq]
        plt.scatter(
            subset['Voltage (kV)'], subset['Cs (nF)'],
            label=f'{freq} Hz', s=35, alpha=0.8,
            color=colours[freq], edgecolors='none'
        )
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    plt.yscale('log')
    
    plt.grid(True, which='major', alpha=0.3, linestyle='-')
    plt.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.xlabel('Peak voltage [kV]', fontsize=12)
    plt.ylabel('Capacitance [nF]', fontsize=12)
    # Removed title: plt.title('Extracted Capacitance Cs (Log Scale)', fontsize=14)
    
    # Modified legend placement - inside plot
    plt.legend(
        loc='upper right',  # Automatic best location inside plot
        title="Freq. Hz", title_fontsize=12, fontsize=12,
        framealpha=0.9  # Slight transparency to not obscure data
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/capacitance_voltage_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Resistance vs. Voltage (Linear scale)
    plt.figure(figsize=(10, 7))
    
    for freq in unique_freqs:
        subset = df[df['Frequency (Hz)'] == freq]
        plt.scatter(
            subset['Voltage (kV)'], subset['Rs (MΩ)'],
            label=f'{freq} Hz', s=35, alpha=0.8,
            color=colours[freq], edgecolors='none'
        )
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    
    # Determine y limits for better visualization
    y_min = 0
    y_max = df['Rs (MΩ)'].max() * 1.1
    plt.ylim(y_min, y_max)
    
    plt.grid(True, alpha=0.3, linestyle='-')
    
    plt.xlabel('Peak voltage [kV]', fontsize=12)
    plt.ylabel('Resistance [MΩ]', fontsize=12)
    # Removed title: plt.title('Extracted Resistance Rs (Linear Scale)', fontsize=14)
    
    # Modified legend placement - inside plot
    plt.legend(
        loc='upper right',  # Typically good for resistance plots
        title="Freq. Hz", title_fontsize=12, fontsize=12,
        framealpha=0.9  # Slight transparency to not obscure data
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/resistance_voltage_linear.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Resistance vs. Voltage (Log scale)
    plt.figure(figsize=(10, 7))
    
    for freq in unique_freqs:
        subset = df[df['Frequency (Hz)'] == freq]
        plt.scatter(
            subset['Voltage (kV)'], subset['Rs (MΩ)'],
            label=f'{freq} Hz', s=35, alpha=0.8,
            color=colours[freq], edgecolors='none'
        )
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    plt.yscale('log')
    
    plt.grid(True, which='major', alpha=0.3, linestyle='-')
    plt.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.xlabel('Peak voltage [kV]', fontsize=12)
    plt.ylabel('Resistance [MΩ]', fontsize=12)
    # Removed title: plt.title('Extracted Resistance Rs (Log Scale)', fontsize=14)
    
    # Modified legend placement - inside plot  
    plt.legend(
        loc='upper right',  # Typically good for log resistance plots
        title="Freq. Hz", title_fontsize=12, fontsize=12,
        framealpha=0.9  # Slight transparency to not obscure data
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/resistance_voltage_log.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_percent_nonlinear(df, output_dir="results_improved"):
    """
    Plot percentage of nonlinear component vs voltage.
    This visualization shows how the MOV becomes increasingly nonlinear
    as voltage increases, which is a key characteristic of varistors.
    """
    if df is None or df.empty:
        print("No data to plot nonlinear percentage.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    
    unique_freqs = sorted(df['Frequency (Hz)'].unique())
    cmap = plt.cm.tab10
    
    for i, freq in enumerate(unique_freqs):
        color = cmap(i % 10)
        subset = df[df['Frequency (Hz)'] == freq]
        
        # Sort by voltage
        subset = subset.sort_values('Voltage (kV)')
        
        # Calculate nonlinear percentage from peaks if not already in data
        if 'nonlinear_percent' not in subset.columns:
            nonlinear_pct = []
            for _, row in subset.iterrows():
                waveform = row['waveform_data']
                if waveform.get('i_linear_peak') is not None and waveform.get('i_res_peak') is not None:
                    i_linear_peak = waveform['i_linear_peak']
                    i_res_peak = waveform['i_res_peak']
                    nonlinear_pct.append((1 - i_linear_peak/i_res_peak) * 100)
                else:
                    nonlinear_pct.append(0)
        else:
            nonlinear_pct = subset['nonlinear_percent'].values
            
        # Plot data points and lines
        plt.plot(subset['Voltage (kV)'], nonlinear_pct, 
                'o-', color=color, linewidth=1.5, markersize=5,
                label=f'{freq} Hz')
    
    plt.xlim(0.5, 12)  # Set consistent x-axis limits
    plt.grid(True, alpha=0.3)
    plt.xlabel('Voltage (kV)', fontsize=12)
    plt.ylabel('Nonlinear Component (%)', fontsize=12)
    # plt.title('Percentage of Nonlinear Current vs Voltage', fontsize=14)
    
    # Set reasonable y-axis limits
    plt.ylim(0, 100)
    
    plt.legend(loc='upper right', fontsize=12)
    
    filename = 'nonlinear_percent_vs_voltage.png'
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300)
    plt.close()
    
    print(f"Nonlinear percentage plot saved to {output_dir}/{filename}")

def plot_vi_characteristics(df_ac, df_dc=None, output_dir="results_improved", alpha=0.05):
    """
    Plot MOV VI Characteristics with clean extrapolation from max resistance point.
    No annotations, just clean data presentation.
    
    Parameters:
    ----------
    df_ac : DataFrame
        DataFrame with AC results
    df_dc : DataFrame
        DataFrame with DC results
    output_dir : str
        Output directory
    alpha : float
        Power law exponent for extrapolation (V ~ I^alpha)
    """
    if (df_ac is None or df_ac.empty) and (df_dc is None or df_dc.empty):
        print("No data to plot VI characteristics.")
        return
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
        
    plt.figure(figsize=(12, 8))
    plt.grid(True, alpha=0.3, linestyle=':')
    
    # Set up color palette
    # Calculate total number of frequencies including DC
    total_freqs = 0
    unique_freqs = []
    if df_ac is not None and not df_ac.empty:
        unique_freqs = sorted(df_ac['Frequency (Hz)'].unique())
        total_freqs = len(unique_freqs)
    
    total_freqs += 1 if df_dc is not None and not df_dc.empty else 0
    colours = plt.cm.tab10(np.linspace(0, 1, max(3, total_freqs)))  # Ensure at least 3 colors
    
    # Set up for AC data
    if df_ac is not None and not df_ac.empty:
        # Plot AC data
        for idx, freq in enumerate(unique_freqs):
            subset = df_ac[df_ac['Frequency (Hz)'] == freq]
            
            if len(subset) < 2:
                plt.scatter(
                    subset['Current (mA)'], subset['Voltage (kV)'],
                    s=35, color=colours[idx], edgecolor='none',
                    alpha=0.8, label=f'{freq} Hz'
                )
                continue
            
            # Sort by current for interpolation
            subset = subset.sort_values('Current (mA)')
            current = subset['Current (mA)'].values
            voltage = subset['Voltage (kV)'].values
            
            if len(current) >= 3:
                try:
                    # Use log-spaced points for interpolation
                    current_interp = np.logspace(
                        np.log10(current.min() * 0.95),
                        np.log10(current.max() * 1.05),
                        200
                    )
                    voltage_interp = pchip_interpolate(current, voltage, current_interp)
                    
                    # Plot the interpolated line
                    plt.plot(
                        current_interp, voltage_interp, '-',
                        linewidth=1.2, color=colours[idx],
                        label=f'{freq} Hz'
                    )
                    # Plot data points on top
                    plt.scatter(
                        current, voltage, s=35, color=colours[idx],
                        edgecolor='none', alpha=0.8
                    )
                except Exception as e:
                    print(f"Error interpolating {freq} Hz data: {e}")
                    plt.scatter(
                        current, voltage, s=35, color=colours[idx],
                        edgecolor='none', alpha=0.8, label=f'{freq} Hz'
                    )
            else:
                # Just two points
                plt.scatter(
                    current, voltage, s=35, color=colours[idx],
                    edgecolor='none', alpha=0.8, label=f'{freq} Hz'
                )
                plt.plot(
                    current, voltage, '-',
                    linewidth=1.2, color=colours[idx]
                )
    
    # Add DC data to the same plot
    if df_dc is not None and not df_dc.empty:
        # Use the last color index for DC data
        dc_color_idx = len(unique_freqs) if df_ac is not None and not df_ac.empty else 0
        
        # Make a copy and sort by voltage
        df_dc_sorted = df_dc.sort_values('Voltage (kV)').copy()
        
        # Find the row with the maximum resistance - this will be our transition point
        max_resistance_idx = df_dc_sorted['Rs (MΩ)'].idxmax()
        max_resistance_row = df_dc_sorted.loc[max_resistance_idx]
        max_resistance = max_resistance_row['Rs (MΩ)']
        transition_voltage = max_resistance_row['Voltage (kV)']
        transition_current = max_resistance_row['Current (mA)']
        
        print(f"Maximum DC resistance: {max_resistance:.2f} MΩ at {transition_voltage:.2f} kV")
        
        # Only use data points with voltage >= transition voltage
        reliable_dc = df_dc_sorted[df_dc_sorted['Voltage (kV)'] >= transition_voltage].copy()
        unreliable_dc = df_dc_sorted[df_dc_sorted['Voltage (kV)'] < transition_voltage].copy()
        
        print(f"Using {len(reliable_dc)} points with voltage >= transition voltage")
        print(f"Excluding {len(unreliable_dc)} points with voltage < transition voltage")
        
        if not reliable_dc.empty:
            # Plot reliable points
            current_reliable = reliable_dc['Current (mA)'].values
            voltage_reliable = reliable_dc['Voltage (kV)'].values
            
            # Plot reliable points and connect with a solid line
            plt.scatter(
                current_reliable, voltage_reliable, s=35, color=colours[dc_color_idx],
                edgecolor='none', alpha=0.9, label='DC'
            )
            plt.plot(
                current_reliable, voltage_reliable, '-',
                linewidth=1.5, color=colours[dc_color_idx]
            )
            
            # Create a trend line for the lower voltage region
            # Use log-spaced currents reaching down to near zero
            min_trend_current = min(0.0001, transition_current * 0.01)
            
            # Create current points extending down from transition point
            trend_currents = np.logspace(
                np.log10(min_trend_current),
                np.log10(transition_current),
                50
            )
            
            # Create voltages using power law V ~ I^alpha
            trend_voltages = transition_voltage * (trend_currents / transition_current) ** alpha
            
            # Make sure trend doesn't go below minimum reasonable voltage
            min_voltage = transition_voltage * 0.5
            trend_voltages = np.maximum(trend_voltages, min_voltage)
            
            # Plot the trend line as dashed - same color, no special markers
            plt.plot(
                trend_currents, trend_voltages, '--',
                linewidth=1.5, color=colours[dc_color_idx],
                alpha=0.8
            )
        else:
            print("Warning: No points found with voltage >= transition voltage")
    
    # Set plot limits based on all data
    all_currents = []
    all_voltages = []
    
    if df_ac is not None and not df_ac.empty:
        all_currents.extend(df_ac['Current (mA)'].values)
        all_voltages.extend(df_ac['Voltage (kV)'].values)
        
    if df_dc is not None and not df_dc.empty and 'reliable_dc' in locals():
        # Only use reliable DC points for axis limits
        all_currents.extend(reliable_dc['Current (mA)'].values)
        all_voltages.extend(reliable_dc['Voltage (kV)'].values)
    
    if all_currents and all_voltages:
        # Set x-axis limits with adaptive minimum
        x_min = min(0.0001, min(all_currents) * 0.1)
        x_max = max(all_currents) * 1.2
        y_min = 0
        y_max = 12  # Set consistent y-axis limit
        
        plt.xscale('log')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
    plt.xlabel('Current (mA)', fontsize=14)
    plt.ylabel('Voltage (kV)', fontsize=14)
    #plt.title('Metal Oxide Varistor (MOV) Vâ€“I Characteristics', fontsize=16)
    plt.legend(fontsize=12, title="Frequency", title_fontsize=13)
    
    plt.grid(True, which='major', alpha=0.3, linestyle='-')
    plt.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    # Slightly thinner axis lines
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vi_characteristics_clean.png', dpi=300)
    plt.close()
    
    print(f"Clean V-I characteristics plot saved to {output_dir}/vi_characteristics_clean.png")

# ===============================================================
# Main Pipeline
# ===============================================================

def main_pipeline(file_pattern="*.csv", output_dir="results_improved"):
    """
    Improved MOV analysis pipeline using integration-based phase shift method
    and reference waveform approach for linear resistive current.
    
    Parameters:
    ----------
    file_pattern : str
        Glob pattern to match files
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Improved MOV Analysis Pipeline...")
    print("Using Integration-Based Phase Shift Method with Reference Waveform Approach")
    
    # Process all files (both AC and DC)
    df_ac, df_dc = process_all_files(file_pattern)
    
    # Process AC data
    if df_ac is not None and not df_ac.empty:
        print("\nProcessing AC measurement data...")
        
        # Convert columns to final units
        # Use standardized frequency value stored during analysis
        df_ac['Frequency (Hz)'] = df_ac['standardized_freq']
        
        # Use peak values
        df_ac['Voltage (kV)'] = df_ac['v_peak'] / 1e3
        
        # Use peak-based resistance as the primary Rs value
        df_ac['Rs (MΩ)'] = df_ac['Rs'] / 1e6
        
        df_ac['Cs (nF)'] = df_ac['Cs'] * 1e9
        df_ac['Current (mA)'] = df_ac['i_peak'] * 1e3
        
        # Sort data
        df_ac_sorted = df_ac.sort_values(by=['Frequency (Hz)', 'Voltage (kV)'])
        
        # Extract linear and non-linear resistance components
        print("\nPerforming linear/non-linear resistance decomposition...")
        df_ac_decomposed = extract_linear_nonlinear_resistance(df_ac_sorted)
        
        # Display final table
        print("\n===== Improved AC Measurements Results =====")
        display_cols = ['filename', 'Voltage (kV)', 'Frequency (Hz)', 
                        'Rs (MΩ)', 'R_linear (MΩ)', 'R_nonlinear (MΩ)',
                        'Cs (nF)', 'Current (mA)']
            
        print(df_ac_decomposed[display_cols].head(10).to_string(index=False))
        print(f"... (total: {len(df_ac_decomposed)} files)")
        
        # Save results
        df_ac_decomposed.to_csv(f'{output_dir}/improved_ac_results.csv', index=False)
        print(f"Saved AC results to {output_dir}/improved_ac_results.csv")
        
        # Generate individual waveform plots for each file
        plot_all_waveforms(df_ac_decomposed, output_dir)
        
        # Generate plots
        print("\nGenerating plots...")
        
        # 1. Capacitance and Resistance plots (linear and log)
        plot_capacitance_resistance(df_ac_decomposed, output_dir)
        
        # 2. Linear resistance vs voltage (both linear and log scale)
        plot_linear_resistance(df_ac_decomposed, output_dir, use_log_scale=False)
        plot_linear_resistance(df_ac_decomposed, output_dir, use_log_scale=True)
        
        # 3. Non-linear resistance vs voltage (both linear and log scale)
        plot_nonlinear_resistance_with_trend(df_ac_decomposed, output_dir, use_log_scale=False)
        plot_nonlinear_resistance_with_trend(df_ac_decomposed, output_dir, use_log_scale=True)

        # demonstrate the physical behavior that as voltage increases, the nonlinear resistance decreases dramatically
        plot_percent_nonlinear(df_ac_decomposed, output_dir)
        
    else:
        print("No AC data available for analysis.")
    
    # Process DC data
    if df_dc is not None and not df_dc.empty:
        print("\nProcessing DC measurement data...")
        
        # Sort by voltage
        df_dc_sorted = df_dc.sort_values(by='Voltage (kV)')
        
        # Display table
        print("\n===== DC Measurements Results =====")
        dc_cols = ['filename', 'Voltage (kV)', 'Current (mA)', 'Rs (MΩ)']
        print(df_dc_sorted[dc_cols].to_string(index=False))
        
        # Save DC results
        df_dc_sorted.to_csv(f'{output_dir}/improved_dc_results.csv', index=False)
        print(f"Saved DC results to {output_dir}/improved_dc_results.csv")
        
        # Generate DC resistance plots
        print("\nGenerating DC resistance plots...")
        plot_dc_resistance(df_dc_sorted, output_dir, use_log_scale=False)
        plot_dc_resistance(df_dc_sorted, output_dir, use_log_scale=True)
        
        # If both AC and DC data available, generate combined plot
        if df_ac is not None and not df_ac.empty:
            print("\nGenerating combined AC and DC resistance plots...")
            plot_combined_resistance(df_ac_decomposed, df_dc_sorted, output_dir, use_log_scale=False)
            plot_combined_resistance(df_ac_decomposed, df_dc_sorted, output_dir, use_log_scale=True)
        
    else:
        print("No DC data available for analysis.")
    
    # Plot Vâ€“I characteristics (combining AC and DC)
    print("\nGenerating V-I characteristics plot...")
    plot_vi_characteristics(
        df_ac_decomposed if df_ac is not None else None, 
        df_dc_sorted if df_dc is not None else None,
        output_dir
    )
    
    print(f"\nAll processing complete! Results saved to {output_dir}/ directory.")

# ===============================================================
# Script Entry
# ===============================================================

if __name__ == '__main__':
    plt.close('all')
    main_pipeline("A5_1_*csv", output_dir="results")