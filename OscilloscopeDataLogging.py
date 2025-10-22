# -*- coding: utf-8 -*-
"""
GDS-1000-U Series Oscilloscope Data Acquisition and Plotting Script

This script connects to a GW Instek GDS-1000-U series oscilloscope,
retrieves the raw waveform data from Channel 1, parses the binary format,
and plots the resulting waveform using Matplotlib.

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
# It might look like 'ASRL/dev/ttyACM0::INSTR' on Linux or 'ASRL3::INSTR' on Windows.
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
    """
    print(f"\nRequesting waveform data from Channel {channel}...")
    try:
        # Command to get the memory data for the specified channel
        instrument.write(f':ACQuire{channel}:MEMory?')

        # --- Robust Two-Step Read for Binary Data ---
        # 1. Read the header to determine the data length.
        # The header is in the format '#NLLLL', where N is the number of length digits,
        # and LLLL is the length of the data block. E.g., '#48008'.
        header_start = instrument.read_bytes(2)
        if header_start[0:1] != b'#':
            raise ConnectionError("Invalid data header from oscilloscope. Expected '#'.")
        
        num_len_digits = int(header_start[1:2].decode('ascii'))
        data_len_str = instrument.read_bytes(num_len_digits).decode('ascii')
        data_len_bytes = int(data_len_str)

        # 2. Read the full data block of the specified length.
        # This block contains: Time Interval (4b) + Channel (1b) + Reserved (3b) + Waveform (8000b)
        binary_data = instrument.read_bytes(data_len_bytes)
        print(f"Successfully received {len(binary_data)} bytes of waveform data.")

        # --- Parse the binary data block ---
        # Unpack the time interval (first 4 bytes, single-precision float, little-endian)
        time_interval = struct.unpack('<f', binary_data[0:4])[0]
        print("Raw time interval bytes:", binary_data[0:4])
        print("Parsed time interval (seconds):", time_interval)
        print(f"Parsed Time Interval: {time_interval * 1e6:.4f} µs/point")
        
        
        # Unpack the waveform data points (starts after 8-byte metadata header)
        waveform_data_start = 8  # 4 (time) + 1 (chan) + 3 (reserved)
        num_points = 4000
        # Format: '>' for big-endian, 'h' for signed short (2 bytes)
        waveform_points = struct.unpack(f'>{num_points}h', binary_data[waveform_data_start:])


        # Cross-check time interval with scope timebase
        tb_scale = float(instrument.query(':TIMebase:SCALe?'))  # seconds/div
        computed_interval = (tb_scale * 16) / num_points  # 10 divisions across the screen
        print(f"Computed interval from timebase: {computed_interval:.6e}s")

        # Prefer the computed interval if they differ by >1%
        if abs(computed_interval - time_interval) / time_interval > 0.01:
            print("Replacing header time interval with computed value.")
            time_interval = computed_interval
            
        # --- Get scaling factors from the oscilloscope for accurate plotting ---
        v_scale = float(instrument.query(f':CHANnel{channel}:SCALe?'))  # Volts/division
        v_offset = float(instrument.query(f':CHANnel{channel}:OFFSet?')) # Vertical offset in Volts

        print(f"Vertical Scale (Volts/Div): {v_scale}")
        print(f"Vertical Offset (Volts): {v_offset}")
        
        # --- Convert raw ADC values to voltage ---
        # The raw 16-bit values need to be scaled to the 8-bit ADC range and then to volts.
        # 256 ADC levels spread over 8 vertical divisions = 32 levels/division.
        adc_values = np.array(waveform_points)
        levels_per_division = 25 
        voltages = (adc_values / levels_per_division) * v_scale + v_offset
        
        
        return voltages, time_interval

    except Exception as e:
        print(f"An error occurred while fetching or parsing data: {e}")
        return None, None

def plot_waveform(voltages, time_interval):
    """Plots the acquired waveform data using Matplotlib."""
    if voltages is None or time_interval is None:
        print("Cannot plot data due to previous errors.")
        return

    num_points = len(voltages)
    # Create the time axis based on the number of points and the interval
    time_axis = np.arange(0, num_points * time_interval, time_interval)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis * 1e3, voltages) # Plot time in milliseconds (ms)
    plt.title('Waveform from GDS-1072-U Oscilloscope')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.xlim(time_axis[0] * 1e3, time_axis[-1] * 1e3)
    plt.show()


def main():
    """Main function to connect, acquire, and plot."""
    # Initialize the VISA resource manager
    rm = pyvisa.ResourceManager()

    # Find and connect to the instrument
    scope = find_instrument(rm, RESOURCE_STRING)

    if scope:
        # Get the data
        voltages, time_interval = get_waveform_data(scope, CHANNEL_TO_QUERY)
        
        # Split voltages into two halves
        mid = len(voltages) // 2
        first_half = voltages[:mid]
        second_half = voltages[mid:]

        vpp_first = first_half.max() - first_half.min()
        vpp_second = second_half.max() - second_half.min()
        print(f"Vpp (first half): {vpp_first:.3f} V")
        print(f"Vpp (second half): {vpp_second:.3f} V")
        
        # Close the connection to the instrument
        scope.close()
        print("\nConnection closed.")

        # Plot the data
        plot_waveform(voltages, time_interval)

if __name__ == "__main__":
    main()

