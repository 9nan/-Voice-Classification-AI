# Perfect Voice Classification Model

A reliable audio classification system that uses ensemble learning to classify audio files while remaining compatible with various audio formats. This model is designed to run efficiently on your CPU and work reliably with your specific audio files.

## Features

- **Reliable Feature Extraction**: Extracts standard audio features including MFCCs, spectral features, energy metrics, and zero crossing rate
- **Ensemble Learning**: Combines multiple machine learning algorithms (Random Forest, Gradient Boosting, SVM) for higher accuracy
- **Automated Optimization**: Uses grid search to find the best parameters for each model
- **Multi-core Processing**: Utilizes all available CPU cores for faster training
- **Detailed Analysis**: Provides comprehensive evaluation metrics and feature importance rankings
- **Format Compatibility**: Works consistently with MP3, WAV, OGG, FLAC, and M4A files

## Setup

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Organize your audio data in the `Data` directory with the following structure:
```
Data/
  ├── class1/
  │   ├── audio1.mp3
  │   ├── audio2.mp3
  │   └── ...
  ├── class2/
  │   ├── audio1.mp3
  │   ├── audio2.mp3
  │   └── ...
  └── ...
```

Each subdirectory in the `Data` folder represents a class category. The name of the directory will be used as the class name.
Supported audio formats: WAV, MP3, OGG, FLAC, M4A.

## Training the Model

Run the training script:

```bash
python simple_voice_classifier.py
```

This will:
1. Extract reliable features from all audio files in the Data directory
2. Split data into training, validation, and test sets
3. Perform grid search to find optimal parameters for multiple models
4. Create an ensemble model that combines the strengths of each classifier
5. Evaluate model performance with detailed metrics
6. Save the trained model to `perfect_voice_model.pkl`

## Making Predictions

After training, you can classify new audio files using:

```bash
python predict.py path/to/audio_file.mp3
```

This will:
1. Extract features from the audio file
2. Make predictions using all models in the ensemble
3. Display the ensemble prediction and confidence score
4. Show individual model predictions for comparison
5. Display probabilities for all possible classes

## Performance

The model focuses on reliable features that work consistently across different audio formats:
- Standard MFCC features (13 coefficients)
- Spectral centroid for timbral characteristics
- Energy metrics for volume/intensity patterns
- Zero crossing rate for fundamental frequency estimation

## Customization

You can customize the model parameters in `simple_voice_classifier.py`:

- `SAMPLE_RATE`: Audio sample rate (default: 22050)
- `DURATION`: Maximum duration in seconds to consider for each audio file (default: 5)
- `DEBUG`: Set to True for detailed error information (default: False)
- Grid search parameters in the `find_optimal_model` function

## Troubleshooting

If you encounter issues:
- Enable DEBUG mode by setting `DEBUG = True` in the script
- Check the error messages for specific problems
- Make sure your audio files are not corrupted
- For extremely large datasets, consider using the `limit_per_class` parameter in the `load_data` function 