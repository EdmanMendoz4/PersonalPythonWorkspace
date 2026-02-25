from dataclasses import dataclass
import matplotlib
import cv2
import pandas as pd
import numpy as np

@dataclass 
class SignalConfig:
    sample_rate: int = 10000
    window_size_sec: float = 0.02
    voltage_min: float = 0.0
    voltage_max: float = 5.0
    
    def __post_init__(self):
        if self.window_size_sec > 1.0:
            # Assuming window_size_sec was inputted in miliseconds instead of seconds.
            self.window_size_sec = self.window_size_sec / 1000.0
        if self.voltage_min >= self.voltage_max:
            raise ValueError("Minimum Voltage must be smaller than Max Voltage")

@dataclass
class VideoConfig:
    video_in: str = "experiment_video.avi"
    video_out: str = "experiment_overlay_final.avi"
    data_csv: str = "experiment_data.csv"      
    timestamps_csv: str = "video_timestamps.csv" 
    
