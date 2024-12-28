# Prosthetic Control Using Machine Learning

This project implements machine learning models for prosthetic control through EEG signal processing and classification. The system is capable of recognizing different hand gestures (rock, paper, scissors, and ok) from EEG data.

## Project Structure

- `smartsvm.py`: Implementation of a Random Forest classifier for hand gesture recognition with t-SNE dimensionality reduction
- `svmmain.py`: Support Vector Machine (SVM) implementation for gesture classification
- `emotionsmanifold.py`: Experimental work with emotion detection from EEG data
- `networkpractice.py`: Neural network practice implementations using TensorFlow

## Features

- Hand gesture recognition for four gestures:
  - Rock
  - Paper
  - Scissors
  - OK
- Data preprocessing and normalization
- Dimensionality reduction using t-SNE
- Multiple classification approaches:
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- Visualization tools:
  - 3D manifold visualization
  - Confusion matrix plots
  - Performance metrics

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Python 3.x

## Data

The project expects EEG data in CSV format with the following structure:
- Multiple input channels (EEG signals)
- Label column (gesture type)
- Data should be organized in separate files for different gestures (0.csv, 1.csv, 2.csv, 3.csv)

## Usage

1. Place your EEG data files in the appropriate directory
2. Run either `smartsvm.py` or `svmmain.py` for gesture classification
3. View the generated visualizations and performance metrics

## Model Performance

The system provides various performance metrics:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- 3D manifold visualization of the gesture clusters

## Future Work

- Integration with real-time EEG data processing
- Implementation of more advanced deep learning models
- Expansion of the gesture recognition set
- Emotion detection integration