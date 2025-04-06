import os
import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import numpy as np
import librosa
from train import extract_features, predict

# Constants
MODEL_PATH = "perfect_voice_model.pkl"

def select_audio_file():
    """Opens a file dialog to select an audio file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac *.m4a")]
    )
    
    return file_path

def check_animal_sound():
    """Main function to select a file and classify it"""
    print("=== DOG OR CAT AUDIO CLASSIFIER ===")
    print("Please select an MP3 file to analyze...")
    
    # Let user select audio file
    audio_file = select_audio_file()
    
    if not audio_file:
        print("No file selected. Exiting.")
        return
    
    print(f"Selected file: {audio_file}")
    print("Analyzing audio...")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        print("Please run train.py first to create the model.")
        return
    
    try:
        # Load model
        model_data = joblib.load(MODEL_PATH)
        
        # Make prediction
        result = predict(audio_file, model_data)
        
        if result is None or isinstance(result, str):
            print(f"Error processing file: {result}")
            return
        
        # Display result
        print("\n=== RESULT ===")
        class_name = result['class']
        confidence = result['confidence'] * 100
        
        # Check if the class is dog or cat
        if class_name.lower() in ['dog', 'cat']:
            print(f"This audio contains a {class_name.upper()} sound!")
        else:
            print(f"This audio contains: {class_name}")
        
        print(f"Confidence: {confidence:.2f}%")
        
        # Show probabilities for all classes
        print("\nProbabilities for all classes:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob*100:.2f}%")
        
        # Show a dialog with the result
        root = tk.Tk()
        root.withdraw()
        
        if class_name.lower() in ['dog', 'cat']:
            message = f"This audio contains a {class_name.upper()} sound!\nConfidence: {confidence:.2f}%"
        else:
            message = f"This audio contains: {class_name}\nConfidence: {confidence:.2f}%"
            
        messagebox.showinfo("Analysis Result", message)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_animal_sound() 