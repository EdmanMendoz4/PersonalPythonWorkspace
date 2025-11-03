# Oscilloscope GUI Project

This project provides a graphical user interface (GUI) for interacting with a GW Instek GDS-1000-U series oscilloscope. The GUI allows users to connect to the oscilloscope, acquire waveform data, and save it to a CSV file. It is designed to be lightweight and overlay other applications without taking up much screen space.

## Project Structure

```
oscilloscope-gui
├── src
│   ├── __init__.py          # Marks the directory as a Python package
│   ├── oscilloscope.py      # Core functionality for oscilloscope communication
│   ├── gui.py               # GUI implementation using Tkinter or PyQt
│   └── terminal.py          # Manages text terminal output in the GUI
├── requirements.txt          # Lists required Python libraries
├── .gitignore                # Specifies files to ignore in Git
└── README.md                 # Documentation for the project
```

## Requirements

To run this project, you need to install the following Python libraries:

- `pyvisa`: For communication with the oscilloscope.
- `numpy`: For numerical operations.
- `matplotlib`: (Optional, if plotting is needed in the future).
- `tkinter` or `PyQt5`: For the GUI framework.

You can install the required libraries using the following command:

```
pip install -r requirements.txt
```

## Usage

1. **Set the Resource String**: Enter the VISA resource string for your oscilloscope in the text entry field. The default value is `ASRL5::INSTR`.

2. **Start/Stop Communication**: Click the button to start or stop communication with the oscilloscope.

3. **Get Waveform Data**: Click the button to acquire waveform data from the oscilloscope and save it to a CSV file.

4. **View Output**: The text terminal area will display output from the print functions, providing feedback on the operations performed.

## Contributing

Contributions to improve the functionality and usability of the project are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.