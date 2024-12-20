# Hand Gesture Recognition System

A computer vision project that recognizes hand gestures using TensorFlow and OpenCV, with optional Raspberry Pi GPIO integration.

## Overview
This project combines deep learning, computer vision, and IoT technologies to create a contactless home control system. Using a camera and trained CNN model, it recognizes eight distinct hand gestures to control different home appliances simulated through GPIO devices on a Raspberry Pi.

### Key Features
- Real-time hand gesture recognition
- CNN-based gesture classification
- GPIO-controlled device simulation
- LCD status display
- Support for both desktop and Raspberry Pi environments
- Custom dataset creation and model training capabilities

## Project Structure

The project contains the following key files:

### Data Collection
- `getimage_real.py`: Captures hand gesture images from webcam for creating the dataset. Images are saved in grayscale format.

### Model Training
- `train_model.py`: Trains the CNN model using TensorFlow/Keras with the collected dataset.
- `tflite_convert.py`: Converts the trained Keras model (.h5) to TensorFlow Lite format for embedded devices.

### Model Evaluation
- `result_test.py`: Evaluates model performance on test dataset, generates classification metrics and confusion matrix.
- `result_validation.py`: Evaluates model performance on validation dataset.
- `test_image.py`: Tests the model on a single image file.

### Implementation
- `verify_webcam.py`: Real-time hand gesture recognition using webcam (desktop version).
- `verify_gpio.py`: Hand gesture recognition with GPIO control (Raspberry Pi version).
- `recognise_gpio_cam.py`: Real-time hand gesture recognition with GPIO control (Raspberry Pi version).
- `recognise_gpio_img.py`: Image-based hand gesture recognition with GPIO control (Raspberry Pi version).


## Setup Instructions
1. Install required packages:

```bash
pip install -r requirements.txt
```

2. For Raspberry Pi GPIO functionality:
- Enable GPIO in Raspberry Pi configuration
- Connect LCD display and GPIO components according to pin configuration
- Run with sudo privileges for GPIO access

## Usage

1. **Collect Dataset**:
```bash
python getimage_real.py
```

2. **Train Model**:
```bash
python train_model.py
```

3. **Run Recognition**:
- For desktop webcam:
```bash
python verify_webcam.py
```

- For Raspberry Pi:
```bash
sudo python recognise_gpio_cam.py
```

## Supported Gestures

The system recognizes 8 different hand gestures:
- FIST
- ONE
- TWO
- THREE
- THUMBSUP
- FIVE
- SIX
- SEVEN

## GPIO Control (Raspberry Pi)

The system controls:
- 2 LEDs
- 1 Buzzer
- 2 Motor endpoints
- LCD Display

Each gesture triggers different combinations of these outputs.

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers
- 2 MaxPooling layers
- Dropout layers
- Dense layers
- Input shape: (64, 64, 1) - Grayscale images
- Output: 8 classes (gestures)

## Dependencies

- TensorFlow
- OpenCV
- NumPy
- Imutils
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- RPi.GPIO (for Raspberry Pi)
- rpi-lcd (for Raspberry Pi)

