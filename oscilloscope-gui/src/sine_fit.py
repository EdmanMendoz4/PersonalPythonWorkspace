import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# --- 1. Define the Sine Function to Fit ---
def sine_func(t, amplitude, omega, phase, offset):
    """
    Defines a sine function for fitting.
    t: time array
    amplitude: V_peak
    omega: angular frequency (2*pi*f)
    phase: phase shift
    offset: DC offset
    """
    return amplitude * np.sin(omega * t + phase) + offset

def fit_waveform(filename, i_peak):
    # --- 3. Load and Prepare Data ---
    readings_dir = Path(r"C:\Users\Sell\Desktop\LecturasOsciloscopio")
    file_path = readings_dir / filename
    print(f"Loading data from {filename}") 
    try:
        df = pd.read_csv(file_path)
        t_data = df['time_s'].values
        v_data = df['voltage_V'].values
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Please check the file.")
        exit()

    # --- 4. Estimate Initial Parameters (p0) for the Fit ---
    # This helps the fitting algorithm converge quickly and accurately.

    # Guess offset from the mean of the data
    p0_offset = np.mean(v_data)

    # Guess amplitude from the data range (ignoring offset)
    p0_amplitude = (np.max(v_data) - np.min(v_data)) / 2.0

    # Guess frequency using FFT (Fast Fourier Transform)
    sample_spacing = np.abs(t_data[1] - t_data[0]) # Avg time between samples
    n = len(v_data)
    # Compute the FFT (ignoring the DC component)
    yf = rfft(v_data - p0_offset) 
    xf = rfftfreq(n, sample_spacing)
    # Find the frequency with the largest amplitude
    dominant_freq_index = np.argmax(np.abs(yf[1:])) + 1
    dominant_freq = xf[dominant_freq_index]
    # Convert to angular frequency (omega = 2*pi*f)
    p0_omega = 2 * np.pi * dominant_freq

    # Guess phase (0 is usually fine)
    p0_phase = 0.0

    # Pack guesses into an array
    p0 = [p0_amplitude, p0_omega, p0_phase, p0_offset]

    # --- 5. Perform the Curve Fit ---
    try:
        # Run the optimization
        params, covariance = curve_fit(sine_func, t_data, v_data, p0=p0, maxfev=10000)

        # Extract fitted parameters
        fit_amplitude = params[0]
        fit_omega = params[1]
        fit_phase = params[2]
        fit_offset = params[3]

        # --- 6. Calculate Results and Display ---
        
        # Get the peak voltage (V_peak) from the fit
        # We use abs() in case the fit finds a negative amplitude (180-deg phase)
        v_peak = np.abs(fit_amplitude)
        
        # Calculate impedance |Z| = V_peak / I_peak
        impedance = v_peak / i_peak
        fit_frequency = fit_omega / (2 * np.pi)

        print("\n--- Fit Results ---")
        print(f"Fitted Amplitude (V_peak): {v_peak:.6f} V")
        print(f"Fitted Frequency: {np.abs(fit_frequency):.6f} Hz")
        print(f"Fitted Offset: {fit_offset:.6f} V")
        print(f"  Impedance |Z| = {impedance:,.2f} Ohms")

        # --- 7. Generate a Plot for Verification ---
        plt.figure(figsize=(12, 6))
        
        # Plot only a subset of data points to keep the plot clean
        plot_skip = 10
        plt.plot(t_data[::plot_skip], v_data[::plot_skip], 'b.', markersize=2, 
                label=f'Oscilloscope Data (1 in {plot_skip} points)')
        
        # Generate a smooth fitted sine wave for plotting
        t_plot = np.linspace(t_data.min(), t_data.max(), 2000)
        v_fit = sine_func(t_plot, *params)
        
        plt.plot(t_plot, v_fit, 'r-', linewidth=2, label='Fitted Sine Wave')
        
        plt.title('Sine Fit to Oscilloscope Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        target_dir = Path(r"C:\Users\Sell\Desktop\LecturasOsciloscopio\Sine_fits")
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = target_dir / f"sine_fit_plot{timestamp}.png"
        plt.savefig(out_path)
        print(f"Verification plot saved to '{f"sine_fit_plot{timestamp}.png"}'")

    except RuntimeError as e:
        print(f"\n--- ERROR ---")
        print(f"Curve fit failed: {e}")
        print("This often happens if the initial guesses (p0) are too far from")
        print("the real values, or if the data is not a clean sine wave.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")