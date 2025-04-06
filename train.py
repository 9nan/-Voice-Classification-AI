import os
import numpy as np
import librosa
import joblib
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count

# Suppress warnings
warnings.filterwarnings("ignore")

# Enhanced configuration
DATA_DIR = "Data"
SAMPLE_RATE = 22050  # Standard sample rate (lower to ensure compatibility)
DURATION = 5  # Slightly longer segments for better features
MODEL_PATH = "perfect_voice_model.pkl"
N_CORES = cpu_count()  # Use all available CPU cores
DEBUG = False  # Set to True for detailed error information

def extract_features(file_path):
    """Extract basic but reliable audio features that work with any file"""
    try:
        # Load audio file with fixed duration
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Check if audio was loaded properly
        if len(y) == 0:
            print(f"Warning: Empty audio file: {file_path}")
            return None
            
        # Pad short audio files
        if len(y) < DURATION * SAMPLE_RATE:
            y = np.pad(y, (0, DURATION * SAMPLE_RATE - len(y)), 'constant')
        
        # Extract features that work reliably with any audio
        features = []
        
        # 1. Basic MFCCs (13 is standard and reliable)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features.extend(mfccs_mean)
        
        # 2. Simple spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroid))
        
        # 3. Energy
        features.append(np.mean(np.abs(y)))
        features.append(np.std(y))
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        
        # Return as numpy array
        if DEBUG:
            print(f"Feature vector shape: {np.array(features).shape}")
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None

def load_data(limit_per_class=None):
    """Load audio files and extract features"""
    X = []
    y = []
    class_indices = {}
    file_paths = []  # Store file paths for later reference
    
    print("Extracting features from audio files...")
    
    # Get all class folders
    class_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    
    for class_idx, class_name in enumerate(class_folders):
        class_indices[class_idx] = class_name
        class_dir = os.path.join(DATA_DIR, class_name)
        print(f"Processing class: {class_name} ({class_idx})")
        
        # Get all audio files in this class folder
        audio_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]
        print(f"Found {len(audio_files)} audio files in {class_name}")
        
        # Limit files per class if specified
        if limit_per_class and len(audio_files) > limit_per_class:
            print(f"Limiting to {limit_per_class} files per class")
            # Sort files to ensure deterministic behavior
            audio_files = sorted(audio_files)[:limit_per_class]
        
        success_count = 0
        for audio_file in tqdm(audio_files):
            file_path = os.path.join(class_dir, audio_file)
            features = extract_features(file_path)
            
            if features is not None:
                X.append(features)
                y.append(class_idx)
                file_paths.append(file_path)
                success_count += 1
        
        print(f"Successfully processed {success_count}/{len(audio_files)} files for class {class_name}")
    
    if len(X) == 0:
        print("No features could be extracted. Check your audio files.")
        return np.array([]), np.array([]), {}, []
    
    print(f"Total extracted features: {len(X)} from {len(class_folders)} classes")
    return np.array(X), np.array(y), class_indices, file_paths

def find_optimal_model(X_train, y_train, X_val, y_val):
    """Find the optimal model through grid search"""
    print("\nFinding optimal model parameters...")
    
    # 1. Random Forest optimization
    print("Optimizing Random Forest...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=N_CORES, class_weight='balanced'),
        rf_params,
        cv=3,
        n_jobs=N_CORES,
        verbose=1
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_score = accuracy_score(y_val, rf_best.predict(X_val))
    print(f"RF best params: {rf_grid.best_params_}")
    print(f"RF validation accuracy: {rf_score:.4f}")
    
    # 2. Gradient Boosting optimization
    print("\nOptimizing Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=3,
        n_jobs=N_CORES,
        verbose=1
    )
    gb_grid.fit(X_train, y_train)
    gb_best = gb_grid.best_estimator_
    gb_score = accuracy_score(y_val, gb_best.predict(X_val))
    print(f"GB best params: {gb_grid.best_params_}")
    print(f"GB validation accuracy: {gb_score:.4f}")
    
    # 3. SVM optimization (only if dataset is small enough)
    if len(X_train) < 5000:  # Only run SVM for smaller datasets
        print("\nOptimizing SVM...")
        svm_params = {
            'C': [1, 10],
            'gamma': ['scale', 'auto']
        }
        
        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=42, kernel='rbf'),
            svm_params,
            cv=3,
            n_jobs=N_CORES,
            verbose=1
        )
        svm_grid.fit(X_train, y_train)
        svm_best = svm_grid.best_estimator_
        svm_score = accuracy_score(y_val, svm_best.predict(X_val))
        print(f"SVM best params: {svm_grid.best_params_}")
        print(f"SVM validation accuracy: {svm_score:.4f}")
    else:
        svm_best = None
        svm_score = 0
        print("Skipping SVM due to large dataset size")
    
    # Create weighted voting ensemble
    models = []
    weights = []
    
    models.append(('rf', rf_best))
    weights.append(rf_score)
    
    models.append(('gb', gb_best))
    weights.append(gb_score)
    
    if svm_best is not None:
        models.append(('svm', svm_best))
        weights.append(svm_score)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        weights=weights
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    return {
        'ensemble': ensemble,
        'rf': rf_best,
        'gb': gb_best,
        'svm': svm_best,
        'weights': weights.tolist()
    }

def train_model():
    """Train an enhanced audio classifier"""
    # Load and extract features
    X, y, class_indices, file_paths = load_data()
    
    if len(X) == 0:
        print("No features extracted. Cannot train model.")
        return
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into train, validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Find the optimal model
    models = find_optimal_model(X_train, y_train, X_val, y_val)
    
    # Train on combined train+val for final evaluation
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    print("\nTraining final model on full training set...")
    models['ensemble'].fit(X_train_full, y_train_full)
    
    # Evaluate on test set
    y_pred = models['ensemble'].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n==== FINAL MODEL PERFORMANCE ====")
    print(f"Ensemble model accuracy: {accuracy:.4f}")
    
    class_names = [class_indices[i] for i in sorted(class_indices.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {i}")
    print(cm)
    
    # Feature importance (from RF model)
    feature_length = X.shape[1]
    feature_names = []
    
    # Create meaningful feature names
    for i in range(13):
        feature_names.append(f"MFCC_{i+1}")
    
    # Add spectral feature names
    spectral_names = [
        "Spectral_Centroid", 
        "Energy_mean", "Energy_std",
        "ZCR_mean"
    ]
    feature_names.extend(spectral_names)
    
    # Ensure feature_names length matches feature vector
    if len(feature_names) > feature_length:
        feature_names = feature_names[:feature_length]
    elif len(feature_names) < feature_length:
        for i in range(len(feature_names), feature_length):
            feature_names.append(f"Feature_{i+1}")
    
    # Get feature importances from Random Forest
    importances = models['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save the model and metadata
    print("\nSaving enhanced model...")
    joblib.dump({
        'ensemble': models['ensemble'],
        'rf': models['rf'],
        'gb': models['gb'],
        'svm': models['svm'],
        'scaler': scaler,
        'class_indices': class_indices,
        'feature_names': feature_names,
        'weights': models['weights'],
        'accuracy': accuracy
    }, MODEL_PATH)
    
    print(f"Perfect model saved to {MODEL_PATH}")
    return models, scaler, class_indices

def predict(audio_file, model_data=None):
    """Make a prediction on a new audio file"""
    if model_data is None:
        # Load the saved model
        try:
            model_data = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error: Could not load model from {MODEL_PATH}: {e}")
            return None
    
    # Extract features from the audio file
    features = extract_features(audio_file)
    
    if features is None:
        return "Error extracting features"
    
    # Scale features
    features_scaled = model_data['scaler'].transform([features])
    
    # Get ensemble and individual models
    ensemble = model_data['ensemble']
    rf = model_data['rf']
    gb = model_data['gb']
    svm = model_data.get('svm', None)  # SVM might not be present
    
    # Make ensemble prediction
    ensemble_prediction = ensemble.predict(features_scaled)[0]
    ensemble_proba = ensemble.predict_proba(features_scaled)[0]
    
    # Get individual model predictions
    individual_predictions = {}
    
    # RF prediction
    rf_prediction = rf.predict(features_scaled)[0]
    rf_proba = rf.predict_proba(features_scaled)[0]
    individual_predictions["Random Forest"] = {
        "class": model_data['class_indices'][rf_prediction],
        "confidence": rf_proba[rf_prediction]
    }
    
    # GB prediction
    gb_prediction = gb.predict(features_scaled)[0]
    gb_proba = gb.predict_proba(features_scaled)[0]
    individual_predictions["Gradient Boosting"] = {
        "class": model_data['class_indices'][gb_prediction],
        "confidence": gb_proba[gb_prediction]
    }
    
    # SVM prediction (if available)
    if svm is not None:
        svm_prediction = svm.predict(features_scaled)[0]
        svm_proba = svm.predict_proba(features_scaled)[0]
        individual_predictions["SVM"] = {
            "class": model_data['class_indices'][svm_prediction],
            "confidence": svm_proba[svm_prediction]
        }
    
    # Get class name
    class_indices = model_data['class_indices']
    class_name = class_indices[ensemble_prediction]
    
    # Create comprehensive result
    result = {
        'class': class_name,
        'confidence': float(ensemble_proba[ensemble_prediction]),
        'probabilities': {class_indices[i]: float(prob) for i, prob in enumerate(ensemble_proba)},
        'individual_models': individual_predictions,
        'model_weights': model_data.get('weights', None)
    }
    
    return result

if __name__ == "__main__":
    print("=== PERFECT VOICE CLASSIFIER ===")
    print("This script will train an enhanced model to classify audio into different categories")
    
    # Check if Data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found!")
        exit(1)
    
    # Check if there are subdirectories in the Data directory
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(subdirs) < 2:
        print(f"Warning: Found only {len(subdirs)} classes in {DATA_DIR}. Need at least 2 classes.")
        if len(subdirs) == 0:
            exit(1)
    
    print(f"Found {len(subdirs)} classes: {', '.join(subdirs)}")
    print(f"Using {N_CORES} CPU cores for optimal processing")
    
    # Debug mode?
    if DEBUG:
        print("DEBUG mode enabled - will show detailed error information")
    
    print("Starting enhanced training process...")
    
    # Train the model
    train_model()
    
    # Example prediction code (uncomment to use)
    # model_data = joblib.load(MODEL_PATH)
    # test_file = "path/to/test/file.mp3"
    # result = predict(test_file, model_data)
    # print(f"Prediction: {result['class']} with {result['confidence']*100:.2f}% confidence") 
