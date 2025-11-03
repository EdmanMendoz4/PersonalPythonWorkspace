# -*- coding: utf-8 -*-
"""
Oscilloscope Communication and Data Retrieval Module

This module contains the core functionality for connecting to a GW Instek GDS-1000-U series oscilloscope,
acquiring waveform data, and saving it to a CSV file. It focuses on communication and data retrieval,
removing plotting and analysis functionalities.

Author: Gemini
Date: 2024-10-21
"""

import pyvisa
import struct
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

# --- Configuration ---
RESOURCE_STRING = 'ASRL5::INSTR'  # Default resource string
CHANNEL_TO_QUERY = 1                # Channel to get data from (1 or 2)
SAMPLES_PER_DIVISION = 25           # oscilloscope: samples (levels) per division

def find_instrument(resource_manager, resource_string):
    """Tries to find and connect to the instrument."""
    try:
        instrument = resource_manager.open_resource(resource_string, timeout=30000)
        instrument.read_termination = '\n'
        instrument.write_termination = '\n'
        idn = instrument.query('*IDN?')
        return instrument
    except pyvisa.errors.VisaIOError as e:
        return None

def get_waveform_data(instrument, channel):
    """Acquires and parses the raw waveform data from the specified channel."""
    try:
        instrument.write(f':ACQuire{channel}:MEMory?')
        header_start = instrument.read_bytes(2)
        if header_start[0:1] != b'#':
            raise ConnectionError("Invalid data header from oscilloscope. Expected '#'.")
        
        num_len_digits = int(header_start[1:2].decode('ascii'))
        data_len_str = instrument.read_bytes(num_len_digits).decode('ascii')
        data_len_bytes = int(data_len_str)
        binary_data = instrument.read_bytes(data_len_bytes)

        waveform_data_start = 8
        num_points = 4000
        waveform_points = struct.unpack(f'>{num_points}h', binary_data[waveform_data_start:])
        
        v_scale = float(instrument.query(f':CHANnel{channel}:SCALe?'))
        v_offset = float(instrument.query(f':CHANnel{channel}:OFFSet?'))
        time_interval = float(instrument.query(':TIMebase:SCALe?'))
        
        adc_values = np.array(waveform_points)
        levels_per_division = SAMPLES_PER_DIVISION 
        voltages = (adc_values / levels_per_division) * v_scale + v_offset
        
        return voltages, time_interval

    except Exception as e:
        return None, None

def save_waveform_csv(voltages, time_interval, filename=None):
    """Save time (s) and voltage (V) columns to a CSV file."""
    if voltages is None or time_interval is None:
        return None

    target_dir = Path(r"C:\Users\Sell\Desktop\LecturasOsciloscopio")
    target_dir.mkdir(parents=True, exist_ok=True)

    num_points = len(voltages)
    # Convert scope time-per-division to per-sample interval by dividing by samples per division
    sample_interval = (time_interval / SAMPLES_PER_DIVISION)/10 # For some reason, there are 250 samples per division
    time_axis = np.arange(0, num_points * sample_interval, sample_interval)[:num_points]

    if filename:
        out_path = Path(filename)
        if not out_path.is_absolute():
            out_path = target_dir / out_path.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = target_dir / f"waveform_{timestamp}.csv"

    try:
        with out_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'voltage_V'])
            for t, v in zip(time_axis, voltages):
                writer.writerow([f"{t:.12e}", f"{v:.12e}"])
        return str(f"waveform_{timestamp}.csv")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        return None

def main(resource_string=RESOURCE_STRING, channel=CHANNEL_TO_QUERY):
    """Main function to connect and acquire data."""
    rm = pyvisa.ResourceManager()
    scope = find_instrument(rm, resource_string)

    if scope:
        voltages, time_interval = get_waveform_data(scope, channel)
        scope.close()
        return voltages, time_interval
    return None, None

if __name__ == "__main__":
    main()