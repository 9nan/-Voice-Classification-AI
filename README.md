# Audio Classification Tool

A simple yet powerful application that analyzes MP3 files to identify the sound content. Using advanced machine learning algorithms, it processes audio features to determine what type of sound is present.

## Features

- User-friendly file selection interface
- Support for multiple audio formats (MP3, WAV, OGG, FLAC, M4A)
- Advanced acoustic feature extraction using librosa
- Ensemble machine learning model for accurate classification
- Detailed results with confidence scores for each class
- Visual feedback through dialog boxes

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
## Usage

1. Run the training script first (if model doesn't exist):
   ```
   python train.py
   ```

2. Run the classification tool:
   ```
   python use.py
   ```

3. Select an audio file when prompted

4. View the classification results both in the terminal and in a popup dialog

## Technical Details

The tool extracts meaningful acoustic features from audio files using:
- MFCCs (Mel-frequency cepstral coefficients)
- Spectral centroids
- Energy measurements
- Zero-crossing rates

Classification is performed using an ensemble of models:
- Random Forest
- Gradient Boosting
- SVM (for smaller datasets)

## Dataset Structure

To train your own models, organize your audio files in the following structure:
```
Data/
  ├── class1/
  │     ├── sample1.mp3
  │     ├── sample2.mp3
  │     └── ...
  ├── class2/
  │     ├── sample1.mp3
  │     ├── sample2.mp3
  │     └── ...
  └── ...
```

Where each folder name in the Data directory represents a class label.

## Requirements

- Python 3.6+
- numpy
- librosa
- scikit-learn
- joblib
- tkinter

## License

This project is open source and available under the MIT License. 
