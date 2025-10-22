# -*- coding: utf-8 -*-
"""
GDS-1000-U Series Oscilloscope Data Acquisition and Plotting Script

This script connects to a GW Instek GDS-1000-U series oscilloscope,
retrieves the raw waveform data from a specified channel, correctly scales it,
and plots the resulting waveform using Matplotlib to mimic the scope's display.

Author: Gemini
Date: 2024-10-21

Prerequisites:
- Python 3.x
- NI-VISA backend installed on your system (e.g., from National Instruments)
- Python libraries: pyvisa, numpy, matplotlib
  Install them using pip:
  pip install -r requirements.txt
"""

import pyvisa
import struct
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# TODO: Replace with your oscilloscope's VISA resource string.
# You can find this using the NI MAX (Measurement & Automation Explorer) software
# or by running `pyvisa.ResourceManager().list_resources()`.
# It might look like 'ASRL/dev/ttyACM0::INSTR' on Linux or 'ASRL5::INSTR' on Windows.
RESOURCE_STRING = 'ASRL5::INSTR'  # <-- IMPORTANT: CHANGE THIS!
CHANNEL_TO_QUERY = 1              # Channel to get data from (1 or 2)

def find_instrument(resource_manager, resource_string):
    """Tries to find and connect to the instrument."""
    try:
        print(f"Attempting to connect to: {resource_string}")
        # Set a longer timeout for the large data transfer.
        # The timeout is in milliseconds. 30000ms = 30s.
        instrument = resource_manager.open_resource(resource_string, timeout=30000)
        instrument.read_termination = '\n'
        instrument.write_termination = '\n'

        # Query the instrument's ID to verify the connection
        idn = instrument.query('*IDN?')
        print(f"Successfully connected to: {idn.strip()}")
        return instrument
    except pyvisa.errors.VisaIOError as e:
        print(f"Error connecting to the instrument: {e}")
        print("\nTroubleshooting tips:")
        print("1. Is the oscilloscope connected to your PC and powered on?")
        print("2. Did you replace 'RESOURCE_STRING' with your device's actual VISA address?")
        print(f"   (Detected resources: {resource_manager.list_resources()})")
        print("3. Is the NI-VISA backend installed correctly?")
        return None

def get_waveform_data(instrument, channel):
    """
    Acquires and parses the raw waveform data from the specified channel.
    This function implements a robust two-step read to handle the definite-length
    binary block sent by the oscilloscope, preventing premature timeouts.
    It also queries the scope for all necessary scaling factors.
    """
    print(f"\nRequesting waveform data from Channel {channel}...")
    try:
        # Command to get the memory data for the specified channel
        instrument.write(f':ACQuire{channel}:MEMory?')

        # --- Robust Two-Step Read for Binary Data ---
        header_start = instrument.read_bytes(2)
        if header_start[0:1] != b'#':
            raise ConnectionError("Invalid data header from oscilloscope. Expected '#'.")
        
        num_len_digits = int(header_start[1:2].decode('ascii'))
        data_len_str = instrument.read_bytes(num_len_digits).decode('ascii')
        data_len_bytes = int(data_len_str)
        
        binary_data = instrument.read_bytes(data_len_bytes)
        print(f"Successfully received {len(binary_data)} bytes of waveform data.")

        # --- Parse the binary data block ---
        time_interval = struct.unpack('<f', binary_data[0:4])[0]
        print(f"Parsed Time Interval: {time_interval * 1e6:.4f} µs/point")
        
        waveform_data_start = 8
        num_points = 4000
        waveform_points = struct.unpack(f'>{num_points}h', binary_data[waveform_data_start:])

        # --- Get scaling and position factors from the oscilloscope ---
        v_scale = float(instrument.query(f':CHANnel{channel}:SCALe?'))
        v_offset = float(instrument.query(f':CHANnel{channel}:OFFSet?'))
        t_scale = float(instrument.query(':TIMebase:SCALe?'))
        t_delay = float(instrument.query(':TIMebase:DELay?'))

        print(f"Vertical Scale (Volts/Div): {v_scale}")
        print(f"Vertical Offset (Volts): {v_offset}")
        print(f"Time Scale (s/Div): {t_scale}")
        print(f"Time Delay (s): {t_delay}")
        
        # =====================================================================
        # VOLTAGE CONVERSION HAPPENS HERE
        # =====================================================================
        # Convert raw ADC integer values to absolute voltage.
        # A positive v_offset on the scope shifts the waveform UP, so we ADD it.
        adc_values = np.array(waveform_points) / 256.0
        levels_per_division = 32.0 
        voltages = (adc_values / levels_per_division) * v_scale + v_offset

        # Return all necessary data and factors for plotting
        return voltages, time_interval, t_scale, v_scale, v_offset, t_delay

    except Exception as e:
        print(f"An error occurred while fetching or parsing data: {e}")
        return None, None, None, None, None, None

def plot_waveform(voltages, time_interval, t_scale, v_scale, v_offset, t_delay):
    """Plots the acquired waveform data using Matplotlib, mimicking the scope screen."""
    if voltages is None:
        print("Cannot plot data due to previous errors.")
        return

    num_points = len(voltages)
    
    # =====================================================================
    # TIME AXIS CONVERSION HAPPENS HERE
    # =====================================================================
    # Create the time axis based on the actual time between points and the
    # horizontal trigger delay, ensuring t=0 is the trigger point.
    total_duration = num_points * time_interval
    t_start = -total_duration / 2 + t_delay
    t_end = total_duration / 2 + t_delay
    time_axis = np.linspace(t_start, t_end, num_points)

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    # Plot the main waveform with a scope-like style
    plt.plot(time_axis * 1e3, voltages, color='#FFF200', linewidth=1.5)

    # --- Configure plot to look like an oscilloscope screen ---
    fig = plt.gcf()
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Set grid and ticks to match the 10x8 divisions
    ax.grid(True, linestyle=':', color='cyan', alpha=0.6)
    
    # Set plot limits based on the scope's settings
    # Horizontal: 10 divisions total, centered on the delay time
    # Vertical: 8 divisions total, centered on the offset
    ax.set_xlim((t_delay - 5 * t_scale) * 1e3, (t_delay + 5 * t_scale) * 1e3)
    ax.set_ylim(v_offset - 4 * v_scale, v_offset + 4 * v_scale)

    # Set labels and title
    ax.set_title('Waveform from GDS-1072-U', color='white', fontsize=16)
    ax.set_xlabel('Time (ms)', color='white', fontsize=12)
    ax.set_ylabel('Voltage (V)', color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    # Add a text box with the scaling factors
    info_text = (f" V/div: {v_scale:.3f}\n"
                 f"T/div: {t_scale*1e3:.3f} ms")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='cyan', alpha=0.7))

    plt.show()


def main():
    """Main function to connect, acquire, and plot."""
    rm = pyvisa.ResourceManager()
    scope = find_instrument(rm, RESOURCE_STRING)

    if scope:
        voltages, time_interval, t_scale, v_scale, v_offset, t_delay = get_waveform_data(scope, CHANNEL_TO_QUERY)
        scope.close()
        print("\nConnection closed.")
        plot_waveform(voltages, time_interval, t_scale, v_scale, v_offset, t_delay)

if __name__ == "__main__":
    main()

