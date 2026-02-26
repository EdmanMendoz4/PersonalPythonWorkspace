import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.stream_readers import AnalogSingleChannelReader
import numpy as np
import msvcrt
import csv
import os
import nidaqmx.system

# Force a hardware reset to clear "Ghost" tasks
print("Resetting Device...")
nidaqmx.system.Device("Dev1").reset_device()
print("Device Reset Complete.")

# --- Configuration ---
CHANNEL_NAME = "Dev1/ai1"       # AI 1 (Differential uses Pin 2 (+) and Pin 3 (-))
SAMPLING_RATE = 10000           # Max rate for USB-6008 (10k)
SAMPLES_PER_CHUNK = 1000        # Read 1000 samples at a time (0.1s latency)
FILENAME = "single_channel_data.csv"

# Remove existing file if it exists to start fresh
if os.path.exists(FILENAME):
    os.remove(FILENAME)

print(f"Configured for {CHANNEL_NAME} at {SAMPLING_RATE} Hz.")
print("Press ENTER to start recording...")

# 1. Wait for Start Trigger (Enter)
while True:
    if msvcrt.kbhit():
        if msvcrt.getch() == b'\r':  # Check for Enter key
            break

print("Recording... Press ESC to stop.")

with nidaqmx.Task() as task:
    # 2. Setup Channel (Differential)
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL_NAME,
        terminal_config=TerminalConfiguration.DIFF,
        min_val=-10.0,
        max_val=10.0
    )

    # 3. Configure Timing (Continuous)
    task.timing.cfg_samp_clk_timing(
        rate=SAMPLING_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=SAMPLES_PER_CHUNK * 10 # Buffer size (optional, defaults usually fine)
    )

    # 4. Setup Single Channel Reader
    # Note: We use AnalogSingleChannelReader for better performance on 1 channel
    reader = AnalogSingleChannelReader(task.in_stream)
    
    # Pre-allocate buffer (1D array for single channel)
    data_buffer = np.zeros(SAMPLES_PER_CHUNK, dtype=np.float64)

    task.start()

    with open(FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Amplitude (V)'])  # Header

        try:
            while True:
                # Read from DAQ buffer into memory
                reader.read_many_sample(
                    data_buffer,
                    number_of_samples_per_channel=SAMPLES_PER_CHUNK
                )

                # Write to CSV
                # reshaping to (-1, 1) creates a column vector for the CSV writer
                writer.writerows(data_buffer.reshape(-1, 1))

                # Check for Stop Trigger (ESC)
                if msvcrt.kbhit():
                    if msvcrt.getch() == b'\x1b':
                        print("\nStop command received.")
                        break
                        
        except Exception as e:
            print(f"\nError: {e}")

print(f"Acquisition complete. Data saved to {FILENAME}")