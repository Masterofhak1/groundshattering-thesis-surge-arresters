#!/usr/bin/env python3
"""
A5 Energy and Power Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, Tuple, List, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
from numba import njit
import warnings

# ===============================================================
# PRESERVED: All Original Utility Functions
# ===============================================================

def find_first_voltage_peak(voltage_signal, time_signal, min_peak_height_ratio=0.3, min_distance_samples=10):
    """
    Find the first significant voltage peak in the signal.
    
    Parameters:
    -----------
    voltage_signal : array_like
        Voltage signal to analyze
    time_signal : array_like
        Corresponding time values
    min_peak_height_ratio : float
        Minimum peak height as ratio of maximum absolute voltage (default: 0.3)
    min_distance_samples : int
        Minimum distance between peaks in samples (default: 10)
    
    Returns:
    --------
    int or None
        Index of first voltage peak, or None if no suitable peak found
    """
    
    # Calculate minimum peak height based on signal amplitude
    max_abs_voltage = np.max(np.abs(voltage_signal))
    min_peak_height = max_abs_voltage * min_peak_height_ratio
    
    print(f"Peak detection parameters:")
    print(f"  Max absolute voltage: {max_abs_voltage:.2f} V")
    print(f"  Minimum peak height: {min_peak_height:.2f} V")
    print(f"  Minimum distance: {min_distance_samples} samples")
    
    # Find positive peaks
    positive_peaks, _ = signal.find_peaks(
        voltage_signal, 
        height=min_peak_height,
        distance=min_distance_samples
    )
    
    # Find negative peaks (for completeness)
    negative_peaks, _ = signal.find_peaks(
        -voltage_signal, 
        height=min_peak_height,
        distance=min_distance_samples
    )
    
    print(f"Found {len(positive_peaks)} positive peaks and {len(negative_peaks)} negative peaks")
    
    if len(positive_peaks) == 0 and len(negative_peaks) == 0:
        print("Warning: No significant peaks found in voltage signal")
        return None
    
    # Find the first peak (positive or negative)
    all_peaks = []
    if len(positive_peaks) > 0:
        all_peaks.extend(positive_peaks.tolist())
    if len(negative_peaks) > 0:
        all_peaks.extend(negative_peaks.tolist())
    
    if not all_peaks:
        return None
    
    first_peak_idx = min(all_peaks)
    peak_time = time_signal[first_peak_idx]
    peak_voltage = voltage_signal[first_peak_idx]
    
    print(f"First voltage peak found:")
    print(f"  Index: {first_peak_idx}")
    print(f"  Time: {peak_time:.4f} s ({peak_time*1000:.1f} ms)")
    print(f"  Voltage: {peak_voltage:.2f} V ({peak_voltage/1000:.3f} kV)")
    
    return first_peak_idx

def standardize_frequency(freq):
    """
    Standardize frequency values to ensure consistent categorization.
    For frequencies near the standard target values, use the exact target value.
    """
    target_freqs = [10, 17, 27, 50, 100, 150, 300, 500]
    for target_freq in target_freqs:
        if abs(freq - target_freq) <= 1.5:
            return target_freq
    return round(freq)

def moving_average(x, window_size):
    """Simple moving average filter."""
    if window_size <= 1:
        return x
    kernel = np.ones(window_size) / float(window_size)
    return np.convolve(x, kernel, mode='same')

def preprocess(x, tx, filename=None, fundamental_freq=None):
    """Preprocess signal using moving average filter."""
    Fs = 1.0 / np.mean(np.diff(tx))
    print(f"Sampling frequency: {Fs:.2f} Hz")
    
    # Default window size (0.0001s)
    window_size = int(round(0.0001 * Fs))
    if window_size < 1:
        window_size = 1
        
    return moving_average(x, window_size)

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

    # Perform hann window convolution
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
    target_freqs = [10, 17, 25, 50, 100, 150, 300, 500]
    for target_freq in target_freqs:
        if abs(f - target_freq) <= 0.5:
            print(f"Frequency close to {target_freq} Hz standard value: {f:.2f} Hz")
    
    # Amplitude correction for Hann window
    window_correction = 2.0
    A = window_correction * M[k] / (N/2)
    
    # Phase with correction for window effect
    P = np.angle(Sp[k]) - delta*np.pi*(N-1)/N
    P = (P + np.pi) % (2*np.pi) - np.pi  # Wrap to ±π
    
    print(f"3-point iDFT with Hann window frequency estimate: {f:.2f} Hz")
    print(f"Phase estimate: {P:.2f} rad, Amplitude: {A:.2f}")
    
    return f, A, P, O

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

def is_dc_file(filename):
    """Check if a file is a DC measurement file."""
    basename = os.path.basename(filename).upper()
    return "DC" in basename

def calculate_atpdraw_tmax(cycles, frequency):
    """
    Calculate Tmax for ATPDraw simulation based on cycles and frequency.
    
    Parameters:
    -----------
    cycles : float
        Number of cycles
    frequency : float
        Frequency in Hz
    
    Returns:
    --------
    float
        Tmax in seconds for ATPDraw simulation
        
    Example:
    --------
    For 10 cycles at 50 Hz:
    T = 1/50 = 0.02 s
    Tmax = 0.02 * 10 = 0.2 s
    """
    period = 1.0 / frequency  # Period of one cycle in seconds
    tmax = period * cycles    # Total time for specified cycles
    return tmax

def load_measurement_file(dataFile):
    """
    Load measurement data file using the same approach as A5 analysis.
    Converts units to SI: Voltage in V, Current in A
    
    Parameters:
    -----------
    dataFile : str
        Path to the data file
        
    Returns:
    --------
    tuple
        (time, voltage, current) or None if failed
    """
    basename = os.path.basename(dataFile)
    print(f"Loading measurement file: {basename}")
    
    try:
        # Try first with faster numpy loading
        try:
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
            numeric_df = df.iloc[numeric_start:].reset_index(drop=True)
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
            numeric_df = numeric_df.dropna()
            data_array = numeric_df.values
        
        if data_array.shape[0] < 100:
            raise ValueError(f"Not enough data points: {data_array.shape[0]} < 100")
        
        # Get data (same format as A5 analysis)
        time_all = data_array[:, 0] * 0.001  # ms to s
        ch1_all = data_array[:, 1]          # Voltage in kV (CH1)
        ch3_all = data_array[:, 2]          # Current in mA (CH3)
        
        # UNIT CONVERSIONS TO SI:
        # Voltage: kV → V (multiply by 1000, then by 3 for amplifier gain)
        # Current: mA → A (multiply by 1e-3)
        ch1_all_proc = preprocess(ch1_all, time_all, basename) * 1000 * 3  # kV → V
        ch3_all_proc = preprocess(ch3_all, time_all, basename) * 1e-3      # mA → A
        
        # Remove DC bias
        ch1_debiased = ch1_all_proc - np.mean(ch1_all_proc)
        ch3_debiased = ch3_all_proc - np.mean(ch3_all_proc)
        
        print(f"Loaded {len(time_all)} samples")
        print(f"Voltage range: {np.min(ch1_debiased):.1f} to {np.max(ch1_debiased):.1f} V")
        print(f"Current range: {np.min(ch3_debiased)*1000:.2f} to {np.max(ch3_debiased)*1000:.2f} mA")
        
        return time_all, ch1_debiased, ch3_debiased
        
    except Exception as e:
        print(f"Error loading {basename}: {str(e)}")
        return None, None, None

# ===============================================================
# PRESERVED: Main Analysis Class
# ===============================================================

class SurgeArresterEnergyAnalyzer:
    """
    PRESERVED: Complete analyzer for surge arrester energy dissipation and power losses.
    All analysis functions remain unchanged.
    """
    
    def __init__(self, dataFile: str, start_from_first_peak: bool = False,
                 skip_initial_samples: int = 0,
                 min_peak_height_ratio: float = 0.3):
        """
        ENHANCED: Initialize analyzer with measurement data file.
        """
        self.dataFile = dataFile
        self.basename = os.path.basename(dataFile)
        self.skip_initial_samples = skip_initial_samples
        self.start_from_first_peak = start_from_first_peak  # NEW
        self.min_peak_height_ratio = min_peak_height_ratio  # NEW
        
        print(f"\nInitializing Enhanced Surge Arrester Energy Analyzer")
        print(f"  File: {self.basename}")
        print(f"  Start method: {'First peak detection' if start_from_first_peak else f'Fixed skip ({skip_initial_samples} samples)'}")  # NEW
        
        self._load_data()
        self._preprocess_data()
        self._estimate_fundamental_frequency()
        self._detect_voltage_cycles()
        self._calculate_atpdraw_tmax()
    
    def _load_data(self):
        """PRESERVED: Load measurement data from file."""
        time, voltage, current = load_measurement_file(self.dataFile)
        
        if time is None:
            raise ValueError(f"Failed to load data from {self.basename}")
        
        self.time_raw = time
        self.voltage_raw = voltage
        self.current_raw = current
        
        print(f"  Raw data loaded successfully")
    
    def _preprocess_data(self):
        """ENHANCED: Preprocess data with intelligent start point selection."""
        # ENHANCED: Determine the optimal starting point
        if self.start_from_first_peak:
            # NEW: Find first significant peak after initial skip
            peak_idx = find_first_voltage_peak(
                self.voltage_raw[self.skip_initial_samples:], 
                self.time_raw[self.skip_initial_samples:],
                self.min_peak_height_ratio
            )
            
            if peak_idx is not None:
                start_idx = self.skip_initial_samples + peak_idx
                print(f"  First significant peak found at index {start_idx}")
            else:
                start_idx = self.skip_initial_samples
                print(f"  No significant peak found, using default skip")
            
            self.analysis_start_idx = start_idx  # NEW: Store for reporting
        else:
            # PRESERVED: Original fixed skip behavior
            start_idx = self.skip_initial_samples
            self.analysis_start_idx = start_idx
        
        # PRESERVED: All original preprocessing logic
        end_idx = len(self.time_raw) - 1
        
        self.time = self.time_raw[start_idx:end_idx+1]
        self.voltage = self.voltage_raw[start_idx:end_idx+1]
        self.current = self.current_raw[start_idx:end_idx+1]
        
        # Calculate sampling parameters
        self.dt = np.mean(np.diff(self.time))
        self.fs = 1.0 / self.dt
        
        print(f"  Sampling frequency: {self.fs:.1f} Hz")
        print(f"  Analysis starts at: {self.time[0]:.6f} s (index {start_idx})")  # ENHANCED
        print(f"  Data coverage: {((end_idx - start_idx + 1) / len(self.time_raw)) * 100:.1f}%")
    
    def _estimate_fundamental_frequency(self):
        """PRESERVED: Estimate fundamental frequency using advanced methods."""
        # Calculate Hann window parameters
        ts = self.time[-1] - self.time[0]  # Total signal duration
        
        # Use iDFT3pHann for frequency estimation
        freq_est, amplitude, phase, offset = iDFT3pHann(self.voltage, ts)
        
        # Standardize frequency
        self.fundamental_freq = freq_est
        self.standardized_freq = standardize_frequency(freq_est)
        
        print(f"  Detected frequency: {self.fundamental_freq:.2f} Hz")
        print(f"  Standardized frequency: {self.standardized_freq} Hz")
    
    def _detect_voltage_cycles(self):
        """PRESERVED: Detect actual number of cycles using voltage zero crossings."""
        # Find rising zero crossings
        zero_crossings = find_zero_crossings(self.time, self.voltage, 'rising')
        
        if len(zero_crossings) < 2:
            # Estimate from duration and frequency
            duration = self.time[-1] - self.time[0]
            self.detected_cycles = duration * self.fundamental_freq
            self.analysis_duration = duration
            print(f"  Insufficient zero crossings, estimated cycles: {self.detected_cycles:.1f}")
        else:
            # Calculate from actual zero crossings
            total_cycle_duration = zero_crossings[-1] - zero_crossings[0]
            self.detected_cycles = len(zero_crossings) - 1
            self.analysis_duration = total_cycle_duration
            print(f"  Detected {self.detected_cycles} complete cycles")
            print(f"  Analysis duration: {self.analysis_duration:.6f} s")
        
        self.zero_crossings = zero_crossings
    
    def _calculate_atpdraw_tmax(self):
        """PRESERVED: Calculate recommended Tmax for ATPDraw simulation."""
        self.tmax_atpdraw = calculate_atpdraw_tmax(self.detected_cycles, self.standardized_freq)
        print(f"  Recommended ATPDraw Tmax: {self.tmax_atpdraw:.4f} s")
    
    def calculate_energy_power(self):
        """
        PRESERVED: Calculate energy and power metrics with correct SI units.
        
        Power calculation: P(t) = V(t) × I(t) [Watts]
        Energy calculation: E(t) = ∫ P(t) dt [Joules]
        """
        print(f"\nCalculating energy and power...")
        print(f"Power calculation: P(t) = V(t)[Volts] × I(t)[Amperes] = P(t)[Watts]")
        print(f"Energy calculation: E(t) = ∫ P(t) dt = E(t)[Joules]")
        
        # Initialize arrays
        n_samples = len(self.time)
        instantaneous_power = np.zeros(n_samples)
        cumulative_energy = np.zeros(n_samples)
        average_power = np.zeros(n_samples)
        
        # Calculate instantaneous power: P(t) = V(t) * I(t)
        instantaneous_power = self.voltage * self.current
        
        # Calculate energy using trapezoidal integration
        cumulative_energy[0] = 0.0
        
        for i in range(1, n_samples):
            dt_step = self.time[i] - self.time[i-1]
            cumulative_energy[i] = cumulative_energy[i-1] + dt_step * (instantaneous_power[i-1] + instantaneous_power[i]) / 2
            
            # Average power: P_avg = E(t) / t (for t > 0)
            time_elapsed = self.time[i] - self.time[0]
            if time_elapsed > 0:
                average_power[i] = cumulative_energy[i] / time_elapsed
            else:
                average_power[i] = 0.0
        
        # Calculate statistics
        power_peak = np.max(np.abs(instantaneous_power))
        power_rms = np.sqrt(np.mean(instantaneous_power**2))
        energy_final = cumulative_energy[-1]
        power_avg_final = average_power[-1]
        energy_per_cycle = energy_final / self.detected_cycles if self.detected_cycles > 0 else 0
        
        # Calculate voltage and current peaks
        v_peak = np.max(np.abs(self.voltage))
        i_peak = np.max(np.abs(self.current))
        
        # Store results
        self.results = {
            'filename': self.basename,
            'start_method': 'first_peak' if self.start_from_first_peak else 'fixed_skip',  # NEW
            'analysis_start_idx': getattr(self, 'analysis_start_idx', self.skip_initial_samples),  # NEW
            'time': self.time,
            'voltage': self.voltage,
            'current': self.current,
            'instantaneous_power': instantaneous_power,
            'cumulative_energy': cumulative_energy,
            'average_power': average_power,
            'fundamental_freq': self.fundamental_freq,
            'standardized_freq': self.standardized_freq,
            'detected_cycles': self.detected_cycles,
            'analysis_duration': self.analysis_duration,
            'tmax_atpdraw': self.tmax_atpdraw,
            'v_peak': v_peak,
            'i_peak': i_peak,
            'power_peak': power_peak,
            'power_rms': power_rms,
            'energy_final': energy_final,
            'power_avg_final': power_avg_final,
            'energy_per_cycle': energy_per_cycle,
            'zero_crossings': self.zero_crossings
        }
        
        print(f"Energy and Power Analysis Results:")
        print(f"  Peak instantaneous power: {power_peak:.6f} W")
        print(f"  RMS power: {power_rms:.6f} W")
        print(f"  Final cumulative energy: {energy_final:.9f} J ({energy_final*1000:.6f} mJ)")
        print(f"  Final average power: {power_avg_final:.6f} W")
        print(f"  Energy per cycle: {energy_per_cycle:.9f} J ({energy_per_cycle*1000:.6f} mJ)")
        print(f"  ATPDraw Tmax for comparison: {self.tmax_atpdraw:.4f} s")
        
        return self.results
    
    def save_results_csv(self, output_dir: str = "energy_results"):
        """PRESERVED: Save complete analysis results to CSV files including ATPDraw Tmax."""
        if not hasattr(self, 'results'):
            print("No results to save. Run calculate_energy_power() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        file_base = os.path.splitext(self.basename)[0]
        
        # Time series data
        time_series_data = pd.DataFrame({
            'Time (s)': self.results['time'],
            'Time (ms)': self.results['time'] * 1000,
            'Voltage (V)': self.results['voltage'],
            'Voltage (kV)': self.results['voltage'] / 1000,
            'Current (A)': self.results['current'],
            'Current (mA)': self.results['current'] * 1000,
            'Instantaneous_Power (W)': self.results['instantaneous_power'],
            'Cumulative_Energy (J)': self.results['cumulative_energy'],
            'Cumulative_Energy (mJ)': self.results['cumulative_energy'] * 1000,
            'Average_Power (W)': self.results['average_power']
        })
        
        time_series_path = os.path.join(output_dir, f"{file_base}_energy_power_timeseries.csv")
        time_series_data.to_csv(time_series_path, index=False)
        
        # ENHANCED: Summary data with start method info
        summary_data = pd.DataFrame([{
            'Filename': self.results['filename'],
            'Start_Method': self.results['start_method'],  # NEW
            'Analysis_Start_Index': self.results['analysis_start_idx'],  # NEW
            'Detected_Freq_Hz': self.results['fundamental_freq'],
            'Standardized_Freq_Hz': self.results['standardized_freq'],
            'Detected_Cycles': self.results['detected_cycles'],
            'Analysis_Duration_s': self.results['analysis_duration'],
            'ATPDraw_Tmax_s': self.results['tmax_atpdraw'],
            'Voltage_Peak_V': self.results['v_peak'],
            'Voltage_Peak_kV': self.results['v_peak'] / 1000,
            'Current_Peak_A': self.results['i_peak'],
            'Current_Peak_mA': self.results['i_peak'] * 1000,
            'Power_Peak_W': self.results['power_peak'],
            'Power_RMS_W': self.results['power_rms'],
            'Energy_Final_J': self.results['energy_final'],
            'Energy_Final_mJ': self.results['energy_final'] * 1000,
            'Power_Avg_Final_W': self.results['power_avg_final'],
            'Energy_per_Cycle_J': self.results['energy_per_cycle'],
            'Energy_per_Cycle_mJ': self.results['energy_per_cycle'] * 1000,
            'Zero_Crossings_Count': len(self.results['zero_crossings'])
        }])
        
        summary_path = os.path.join(output_dir, f"{file_base}_energy_power_summary.csv")
        summary_data.to_csv(summary_path, index=False)
        
        print(f"Results saved:")
        print(f"  Time series: {time_series_path}")
        print(f"  Summary: {summary_path}")
        
        return time_series_path, summary_path
    
    def plot_energy_power_analysis(self, output_dir: str = "energy_results"):
        """MODIFIED: Create energy and power analysis plot matching FD model style."""
        if not hasattr(self, 'results'):
            print("No results to plot. Run calculate_energy_power() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        file_base = os.path.splitext(self.basename)[0]
        
        time_ms = self.results['time'] * 1000  # Convert to milliseconds
        
        # Create figure with 2 subplots (matching FD model layout)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top subplot: Voltage and Current
        ax1_twin = ax1.twinx()
        
        # Voltage plot (blue, left axis) - PRESERVING ORIGINAL COLOR
        ax1.plot(time_ms, self.results['voltage']/1000, 'b-', linewidth=1.5, label='Voltage (kV)')
        ax1.set_ylabel('Voltage (kV)', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Current plot (orange, right axis) - PRESERVING ORIGINAL COLOR
        ax1_twin.plot(time_ms, self.results['current']*1000, 'orange', linewidth=1.5, label='Current (mA)')
        ax1_twin.set_ylabel('Current (mA)', color='orange', fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor='orange')
        
        # Mark analysis start if using peak detection
        if self.start_from_first_peak:
            ax1.axvline(x=time_ms[0], color='red', linestyle='--', alpha=0.7, linewidth=1)
            start_method_str = " (Peak Start)"
        else:
            start_method_str = ""
        
        ax1.set_title(f'Surge Arrester Waveforms - {self.results["detected_cycles"]:.1f} Cycles{start_method_str}\n'
                     f'Peak V: {self.results["v_peak"]/1000:.2f} kV, Peak I: {self.results["i_peak"]*1000:.1f} mA, '
                     f'Freq: {self.results["standardized_freq"]:.0f} Hz, ATPDraw Tmax: {self.results["tmax_atpdraw"]:.4f} s\n'
                     f'{self.basename}', fontsize=12)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Bottom subplot: Power and Energy
        ax2_twin = ax2.twinx()
        
        # Power plots (left axis) - PRESERVING ORIGINAL COLORS
        ax2.plot(time_ms, self.results['instantaneous_power'], 'g-', linewidth=1.0, 
                 label='Instantaneous Power', alpha=0.8)
        ax2.plot(time_ms, self.results['average_power'], 'b-', linewidth=2.0, 
                 label='Average Power')
        
        ax2.set_ylabel('Power (W)', color='k', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='k')
        ax2.grid(True, alpha=0.3)
        
        # Mark analysis start on power plot too
        if self.start_from_first_peak:
            ax2.axvline(x=time_ms[0], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Energy plot (right axis, red) - PRESERVING ORIGINAL COLOR
        ax2_twin.plot(time_ms, self.results['cumulative_energy'], 'r-', linewidth=2.0, 
                      label='Cumulative Energy')
        ax2_twin.set_ylabel('Energy (J)', color='r', fontsize=12)
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_title(f'Power and Energy Analysis\n'
                     f'Final Energy: {self.results["energy_final"]*1000:.3f} mJ, '
                     f'Energy/Cycle: {self.results["energy_per_cycle"]*1000:.3f} mJ, '
                     f'Avg Power: {self.results["power_avg_final"]:.3f} W', fontsize=12)
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        method_suffix = "peak_start" if self.start_from_first_peak else "fixed_start"
        plot_path = os.path.join(output_dir, f"{file_base}_energy_power_analysis_{method_suffix}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_path}")
        return plot_path

# ===============================================================
# PRESERVED: All Original Analysis Functions
# ===============================================================

def analyze_single_file(dataFile: str, output_dir: str = "energy_results", 
                       start_from_first_peak: bool = False,
                       skip_initial_samples: int = 0,
                       min_peak_height_ratio: float = 0.3) -> Optional[Dict]:
    """
    ENHANCED: Analyze a single AC measurement file with optional peak start.
    
    Parameters:
    -----------
    dataFile : str
        Path to the measurement file
    output_dir : str
        Output directory for results
    start_from_first_peak : bool
        NEW: If True, start analysis from first voltage peak
    skip_initial_samples : int
        Number of initial samples to skip
    min_peak_height_ratio : float
        NEW: Minimum peak height ratio for peak detection
    
    Returns:
    --------
    Dict or None
        Analysis results dictionary or None if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {os.path.basename(dataFile)}")
        print(f"{'='*60}")
        
        # ENHANCED: Create analyzer with peak start option
        analyzer = SurgeArresterEnergyAnalyzer(
            dataFile, 
            start_from_first_peak=start_from_first_peak,  # NEW
            skip_initial_samples=skip_initial_samples,
            min_peak_height_ratio=min_peak_height_ratio  # NEW
        )
        
        # Calculate energy and power
        results = analyzer.calculate_energy_power()
        
        # Save results
        analyzer.save_results_csv(output_dir)
        analyzer.plot_energy_power_analysis(output_dir)
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {os.path.basename(dataFile)}: {str(e)}")
        return None

def batch_analyze(file_pattern: str = "A5_1_*.csv", 
                 output_dir: str = "energy_results",
                 start_from_first_peak: bool = False,
                 skip_initial_samples: int = 0,
                 min_peak_height_ratio: float = 0.3) -> pd.DataFrame:
    """
    ENHANCED: Batch analyze multiple AC measurement files with optional peak start.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern for input files
    output_dir : str
        Output directory
    start_from_first_peak : bool
        NEW: If True, start analysis from first voltage peak (default: False)
    skip_initial_samples : int
        Number of initial samples to skip (default: 0)
    min_peak_height_ratio : float
        NEW: Minimum peak height ratio for peak detection (default: 0.3)
        
    Returns:
    --------
    DataFrame
        Comprehensive summary of all analyses including ATPDraw Tmax
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} files to process")
    
    # Filter out DC files
    ac_files = [f for f in files if not is_dc_file(f)]
    dc_files = [f for f in files if is_dc_file(f)]
    
    print(f"AC files to analyze: {len(ac_files)}")
    print(f"DC files skipped: {len(dc_files)}")
    
    if dc_files:
        print("Skipped DC files:")
        for dc_file in dc_files:
            print(f"  - {os.path.basename(dc_file)}")
    
    all_summaries = []
    
    for file_path in ac_files:
        # ENHANCED: Pass peak start parameters
        results = analyze_single_file(
            file_path, 
            output_dir, 
            start_from_first_peak=start_from_first_peak,  # NEW
            skip_initial_samples=skip_initial_samples,
            min_peak_height_ratio=min_peak_height_ratio  # NEW
        )
        
        if results:
            # ENHANCED: Create comprehensive summary entry with start method info
            summary = {
                'Filename': results['filename'],
                'Start_Method': results['start_method'],  # NEW
                'Analysis_Start_Index': results['analysis_start_idx'],  # NEW
                'Detected_Freq_Hz': results['fundamental_freq'],
                'Standardized_Freq_Hz': results['standardized_freq'],
                'Detected_Cycles': results['detected_cycles'],
                'Analysis_Duration_s': results['analysis_duration'],
                'ATPDraw_Tmax_s': results['tmax_atpdraw'],
                'Voltage_Peak_V': results['v_peak'],
                'Voltage_Peak_kV': results['v_peak'] / 1000,
                'Current_Peak_A': results['i_peak'],
                'Current_Peak_mA': results['i_peak'] * 1000,
                'Power_Peak_W': results['power_peak'],
                'Power_RMS_W': results['power_rms'],
                'Energy_Final_J': results['energy_final'],
                'Energy_Final_mJ': results['energy_final'] * 1000,
                'Power_Avg_Final_W': results['power_avg_final'],
                'Energy_per_Cycle_J': results['energy_per_cycle'],
                'Energy_per_Cycle_mJ': results['energy_per_cycle'] * 1000,
                'Zero_Crossings_Count': len(results['zero_crossings'])
            }
            all_summaries.append(summary)
    
    # Create comprehensive summary DataFrame
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        
        # Sort by standardized frequency and voltage
        summary_df = summary_df.sort_values(['Standardized_Freq_Hz', 'Voltage_Peak_kV'])
        
        # ENHANCED: Save batch summary with method info
        os.makedirs(output_dir, exist_ok=True)
        method_suffix = "peak_start" if start_from_first_peak else "fixed_start"
        batch_summary_path = os.path.join(output_dir, f"batch_energy_analysis_summary_{method_suffix}.csv")
        summary_df.to_csv(batch_summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Analysis method: {'First voltage peak' if start_from_first_peak else 'Fixed sample skip'}")  # NEW
        print(f"Successfully analyzed: {len(all_summaries)} AC files")
        print(f"Skipped: {len(dc_files)} DC files")
        print(f"Results saved to: {batch_summary_path}")
        
        # PRESERVED: Display summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Average detected cycles: {summary_df['Detected_Cycles'].mean():.2f}")
        print(f"  Cycle count range: {summary_df['Detected_Cycles'].min():.1f} - {summary_df['Detected_Cycles'].max():.1f}")
        print(f"  Average ATPDraw Tmax: {summary_df['ATPDraw_Tmax_s'].mean():.4f} s")
        print(f"  Tmax range: {summary_df['ATPDraw_Tmax_s'].min():.4f} - {summary_df['ATPDraw_Tmax_s'].max():.4f} s")
        print(f"  Average energy per measurement: {summary_df['Energy_Final_J'].mean():.6f} J")
        print(f"  Energy range: {summary_df['Energy_Final_J'].min():.6f} - {summary_df['Energy_Final_J'].max():.6f} J")
        print(f"  Average power: {summary_df['Power_Avg_Final_W'].mean():.6f} W")
        
        return summary_df
    else:
        print("No AC files were successfully analyzed")
        return pd.DataFrame()

# ===============================================================
# MODIFIED: Comparison Plots (FD Model Style with Original Colors)
# ===============================================================

def create_enhanced_batch_comparison_plots(summary_df: pd.DataFrame, output_dir: str = "energy_results"):
    """
    Create separate plots for each analysis type.
    """
    if summary_df is None or summary_df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # PRESERVED: Original color palette from A5
    unique_freqs = sorted(summary_df['Standardized_Freq_Hz'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_freqs)))
    
    # Plot 1: Energy vs Voltage
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, freq in enumerate(unique_freqs):
        freq_data = summary_df[summary_df['Standardized_Freq_Hz'] == freq].sort_values('Voltage_Peak_kV')
        ax1.plot(freq_data['Voltage_Peak_kV'], freq_data['Energy_Final_mJ'], 
                'o-', color=colors[i], label=f'{freq:.0f} Hz', markersize=8, linewidth=2)
    
    ax1.set_xlabel('Peak Voltage (kV)', fontsize=12)
    ax1.set_ylabel('Final Energy (mJ)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A5_energy_vs_voltage_peak_start.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Power vs Voltage
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, freq in enumerate(unique_freqs):
        freq_data = summary_df[summary_df['Standardized_Freq_Hz'] == freq].sort_values('Voltage_Peak_kV')
        ax2.plot(freq_data['Voltage_Peak_kV'], freq_data['Power_Avg_Final_W'], 
                'o-', color=colors[i], label=f'{freq:.0f} Hz', markersize=8, linewidth=2)
    
    ax2.set_xlabel('Peak Voltage (kV)', fontsize=12)
    ax2.set_ylabel('Average Power (W)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A5_power_vs_voltage_peak_start.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Cycle-Normalized Energy
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for i, freq in enumerate(unique_freqs):
        freq_data = summary_df[summary_df['Standardized_Freq_Hz'] == freq].sort_values('Voltage_Peak_kV')
        ax3.plot(freq_data['Voltage_Peak_kV'], freq_data['Energy_per_Cycle_mJ'], 
                'o-', color=colors[i], label=f'{freq:.0f} Hz', markersize=8, linewidth=2)
    
    ax3.set_xlabel('Peak Voltage (kV)', fontsize=12)
    ax3.set_ylabel('Energy per Cycle (mJ)', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A5_cycle_normalized_energy_peak_start.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Frequency Response at Different Voltages
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    voltage_levels = sorted(summary_df['Voltage_Peak_kV'].unique())
    voltage_colors = plt.cm.plasma(np.linspace(0, 1, len(voltage_levels)))
    
    for j, voltage in enumerate(voltage_levels):
        voltage_data = summary_df[summary_df['Voltage_Peak_kV'] == voltage].sort_values('Standardized_Freq_Hz')
        if len(voltage_data) >= 2:
            ax4.plot(voltage_data['Standardized_Freq_Hz'], voltage_data['Energy_Final_mJ'], 
                    'o-', color=voltage_colors[j], label=f'{voltage:.1f} kV', 
                    markersize=8, linewidth=2)
    
    ax4.set_xlabel('Frequency (Hz)', fontsize=12)
    ax4.set_ylabel('Final Energy (mJ)', fontsize=12)
    ax4.legend(loc='best', fontsize=10, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "A5_frequency_response_peak_start.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced batch comparison plots saved as separate files in {output_dir}")

# ===============================================================
# PRESERVED: Main Function
# ===============================================================

def main_energy_analysis(file_pattern: str = "A5_1_*.csv", 
                        output_dir: str = "energy_results",
                        start_from_first_peak: bool = False,
                        skip_initial_samples: int = 0,
                        min_peak_height_ratio: float = 0.3):
    """
    ENHANCED: Main function for surge arrester energy and power analysis with optional peak start.
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern for input files
    output_dir : str
        Output directory
    start_from_first_peak : bool
        NEW: If True, start analysis from first voltage peak (default: False)
    skip_initial_samples : int
        Number of initial samples to skip (default: 0)
    min_peak_height_ratio : float
        NEW: Minimum peak height ratio for peak detection (default: 0.3)
    """
    print("Enhanced Surge Arrester Energy and Power Analysis")
    print("High Voltage Electrical Engineering")
    print("="*60)
    print("NEW FEATURE: Optional start from first voltage peak for better cycle alignment")
    print("PRESERVED: All original comprehensive analysis capabilities")
    print(f"Analysis method: {'First voltage peak' if start_from_first_peak else 'Fixed sample skip'}")
    if start_from_first_peak:
        print(f"Peak detection threshold: {min_peak_height_ratio*100:.0f}% of max voltage")
    print("AC measurements only - DC files excluded")
    print("Uses standardized frequencies for consistent grouping")
    print("Includes ATPDraw Tmax calculation for simulation comparison")
    print(f"Initial samples skipped: {skip_initial_samples}")
    print("Proper SI unit conversions: P[W] = V[V] × I[A], E[J] = ∫P dt")
    print("="*60)
    
    # ENHANCED: Run batch analysis with peak start option
    summary_df = batch_analyze(
        file_pattern, 
        output_dir, 
        start_from_first_peak=start_from_first_peak,
        skip_initial_samples=skip_initial_samples,
        min_peak_height_ratio=min_peak_height_ratio
    )
    
    # MODIFIED: Create comparison plots with FD model style
    if not summary_df.empty:
        create_enhanced_batch_comparison_plots(summary_df, output_dir)
    
    print(f"\nAnalysis complete! All results available in: {output_dir}/")
    print("\nKey output files:")
    method_suffix = "peak_start" if start_from_first_peak else "fixed_start"
    print(f"  - batch_energy_analysis_summary_{method_suffix}.csv (main results)")
    print(f"  - enhanced_batch_comparison_analysis_{method_suffix}.png (visualization)")
    print("  - Individual file results: *_energy_power_summary.csv")
    print("  - Time series data: *_energy_power_timeseries.csv")
    print(f"  - Analysis plots: *_energy_power_analysis_{method_suffix}.png")
    
    return summary_df

# ===============================================================
# Script Execution
# ===============================================================

if __name__ == "__main__":
    # Example usage with peak start option
    results_df = main_energy_analysis(
        file_pattern="A5_1_*.csv",
        output_dir="energy_results_peak_start",
        start_from_first_peak=True,      # NEW: Start from first voltage peak
        skip_initial_samples=10,        # dont skip initial transients first
        min_peak_height_ratio=0.3        # NEW: Peak detection sensitivity
    )