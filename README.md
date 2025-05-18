# Throw Fire

An interactive AR application that lets you create and throw fire with hand gestures!

## Features

- Create small fires by pinching your thumb and index finger together
- Collect all fires by making a fist
- Throw collected fires by opening your palm toward the camera

## Project Versions

This project has two versions:

1. **Full Version (`throw_fire.py`)**: Uses MediaPipe for hand gesture recognition.
2. **Simple Version (`simple_fire.py`)**: Uses only OpenCV and mouse controls instead of hand gestures.

## Requirements

- Python 3.7+
- Webcam
- For full version: The packages listed in `requirements.txt`
- For simple version: Only OpenCV and NumPy

## Installation

### Windows

1. Clone or download this repository
2. Open PowerShell or Command Prompt
3. Navigate to the project directory
4. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Manual Installation

If you encounter issues with the requirements file, you can install the packages individually:

```
pip install opencv-python
pip install mediapipe
pip install numpy
```

For the simple version, you only need:
```
pip install opencv-python
pip install numpy
```

### Other OS

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Full Version (Hand Gestures)

Run the full application:

```
python throw_fire.py
```

#### Gestures

1. **Create Fire**: Pinch your thumb and index finger together
2. **Collect Fires**: Make a fist after creating fires
3. **Throw Fire**: After collecting fires, open your palm facing the camera

### Simple Version (Mouse Controls)

If you're having trouble with MediaPipe installation or hand gesture recognition, try the simplified version:

```
python simple_fire.py
```

#### Mouse Controls

1. **Create Fire**: Left click anywhere on the screen
2. **Collect Fires**: Right click to collect all fires
3. **Throw Fire**: Middle click to throw collected fires

### General Controls

Press 'q' to quit either application.

## How It Works

This application uses:
- OpenCV for webcam access and image processing
- MediaPipe for hand tracking and gesture recognition (full version only)
- NumPy for numerical operations

The program tracks user input (hand gestures or mouse clicks), and creates visual fire effects accordingly.

## Customization

See [CUSTOMIZATION.md](CUSTOMIZATION.md) for details on how to:
- Change fire colors and appearance
- Adjust animation parameters
- Modify gesture recognition sensitivity
- Add sound effects and other enhancements

## Troubleshooting

### Installation Issues
- If pip fails to install packages, try installing them one by one as shown in the Manual Installation section
- Make sure you have an internet connection when installing packages
- If OpenCV installation fails, try: `pip install opencv-contrib-python` instead
- For MediaPipe installation issues, check their [official documentation](https://google.github.io/mediapipe/getting_started/install.html)
- If you're still having issues, try the simple version which only requires OpenCV

### Runtime Issues
- If the camera doesn't start, make sure no other application is using it
- For best hand tracking, ensure good lighting conditions
- If gestures aren't detected accurately, try adjusting your hand position or distance from the camera
- On Windows, if you get errors about missing DLLs, you may need to install the Microsoft Visual C++ Redistributable package 