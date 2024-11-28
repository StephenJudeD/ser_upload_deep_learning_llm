Audio Emotion Recognition Project
=================================

Introduction
------------

The Audio Emotion Recognition Project aims to analyze audio recordings to detect and interpret the emotions expressed within them. This project leverages state-of-the-art machine learning techniques and a variety of datasets to build robust models capable of understanding emotional nuances in speech.

The journey of this project begins with data collection and preprocessing, leading to feature extraction and augmentation techniques, which culminate in the development of a machine learning model that can accurately classify emotions in audio data.

SER Datasets: Training Data
--------

The foundation of this project lies in the diverse datasets used for training our models. The following datasets are utilized:

-   RAVDESS: The Ryerson Audio-Visual Database of Emotional Speech and Song provides a rich array of emotional speech and song recordings, which serve as a primary resource for training emotion recognition models.
-   SAVEE: The Surrey Audio-Visual Expressed Emotion dataset includes recordings of male actors expressing different emotions, contributing to the model's understanding of emotional variation in male voices.
-   TESS: The Toronto Emotional Speech Set features emotional recordings from female actors, helping balance the dataset and enhancing the model's capability to recognize emotions across genders. https://drive.google.com/drive/folders/1qopqXkRnYUmOmW07DbetPDiacEfFg7ZW?usp=sharing
-   CREMA-D: A crowd-sourced emotional multimodal actors dataset that includes a wide range of emotions, providing additional context and diversity in emotional expression. https://drive.google.com/drive/folders/16D_EvScAQN7zG8Mo-igSnSEj_ftHvHpC?usp=sharing

SER Datasets: Test Data for Cross Corpora Validation
------------------------

To ensure the models can generalize well across different datasets, we employ cross corpora validation. This technique evaluates the model's performance on unseen data from different datasets, thus assessing its robustness and adaptability. 

-   Emotion Speech Dataset: ESD is an Emotional Speech Database for voice conversion research. The ESD database consists of 350 parallel utterances spoken by 10 native English Speakers:  https://drive.google.com/drive/folders/1a8tBh9d95GM_84TaZ9o4OkIWa0SIrdgV?usp=sharing
-   EmoDB: The Emotional Speech Database (German) adds further linguistic diversity and emotional contexts, allowing the model to generalize better across different languages and cultural expressions. https://drive.google.com/drive/folders/1Q4luLXPGzpM79kjex5VORk5arCzCr3jN?usp=sharing

Real World Speech Data
------------------------
Snippets from Speech Data to test in real world Scenarios

-  https://drive.google.com/drive/folders/14BU4i4zAg_XmF6lPK83EWwNhATpc1r3v?usp=sharing


Feature Extraction
------------------

The feature extraction process is crucial for speech emotion recognition (SER). This process involves several key steps designed to transform raw audio signals into meaningful features that can be utilized by machine learning models:

-   MFCCs (Mel-Frequency Cepstral Coefficients): These coefficients represent the short-term power spectrum of sound, capturing the timbral characteristics of the audio.
-   Mel Spectrograms: This visual representation of the spectrum of frequencies in sound provides time-frequency information that is essential for emotion classification.
-   Spectral Centroid: This feature indicates where the center of mass of the spectrum is located, often correlating with the perceived brightness of the audio.
-   Chroma Features: These features represent the energy distribution of pitches in the audio, reflecting harmonic and melodic characteristics.
-   Zero-Crossing Rate: This feature measures the rate at which the audio signal changes from positive to negative, providing insights into the signal's frequency content.

This comprehensive feature extraction helps the models learn from diverse audio conditions, improving their robustness and accuracy.

Audio Augmentation
------------------

The audio augmentation process is essential for enhancing the robustness of the emotion recognition models. The following methods are employed to augment the audio data:

-   Adding Background Noise: This technique involves introducing a controlled level of random noise to the audio samples. It helps simulate real-world conditions where audio recordings might contain unwanted sounds, making the models more resilient to variations in the audio environment.
-   Time Stretching: This method alters the speed of the audio without changing its pitch. By stretching or compressing the audio duration, we create variations in the speech tempo, allowing the models to learn from different speaking rates and improve their adaptability to real-life scenarios.
-   Pitch Shifting: This technique adjusts the pitch of the audio while maintaining the same speed. Pitch shifting provides the model with diverse tonal variations, enabling it to recognize emotions across different vocal pitches, which can be particularly relevant when analyzing emotions expressed by different speakers.
-   Shifting: Audio shifting involves moving the audio samples forward or backward in time. This can help the model learn to recognize emotions even when parts of the audio are truncated or when the beginning of the speech might be missing.
-   Adding Echo and Reverb Effects: By applying echo and reverb effects to the audio, we simulate different acoustic environments. These effects can help the model generalize better by learning to identify emotions in recordings that may be affected by different spatial characteristics.

Models Overview
---------------

After the audio data has been prepared through feature extraction and augmentation, several machine learning models are trained to classify emotions. Below are the details of the models developed:

-   Model 1: CNN + Bidirectional LSTM + Bidirectional GRU with Attention

    -   Final Accuracy: 0.9650
    -   Final Validation Accuracy: 0.9085
    -   Final Loss: 0.0957
    -   Final Validation Loss: 0.3040
    -   Evaluation: This model excels in capturing both spatial and temporal features, making it highly effective for complex emotional patterns in speech. The attention mechanism enhances its ability to focus on significant features, resulting in high accuracy and validation performance.
-   Model 2: CNN Only

    -   Final Accuracy: 0.9727
    -   Final Validation Accuracy: 0.8793
    -   Final Loss: 0.0761
    -   Final Validation Loss: 0.4433
    -   Evaluation: The CNN-only model achieves high accuracy, indicating effective feature extraction. However, its lower validation accuracy suggests it may not generalize as well to unseen data, making it suitable for simpler datasets or tasks.
-   Model 3: CNN + Bidirectional LSTM

    -   Final Accuracy: 0.9751
    -   Final Validation Accuracy: 0.9036
    -   Final Loss: 0.0703
    -   Final Validation Loss: 0.3388
    -   Evaluation: This model effectively combines CNNs with LSTMs, capturing both spatial and temporal features. It shows strong validation accuracy, indicating good generalization. The absence of GRUs and attention may limit its performance on more complex datasets compared to Model 1.
-   Model 4: CNN + GRU

    -   Final Accuracy: 0.9714
    -   Final Validation Accuracy: 0.9012
    -   Final Loss: 0.0817
    -   Final Validation Loss: 0.3441
    -   Evaluation: The CNN + GRU model performs well, with validation accuracy similar to Model 3. However, it lacks the attention mechanism, which may limit its ability to focus on important features in the data. This model is effective for datasets where GRUs can capture the necessary temporal dynamics.
-   Model 5: CNN + Multi-Head Attention + Bidirectional LSTM + Bidirectional GRU

    -   Final Accuracy: 0.9757
    -   Final Validation Accuracy: 0.8980
    -   Final Loss: 0.0666
    -   Final Validation Loss: 0.3886
    -   Evaluation: This model incorporates multi-head attention, allowing it to focus on multiple aspects of the input sequence simultaneously. While it achieves high accuracy, its validation accuracy is slightly lower than Model 3, suggesting a potential risk of overfitting. It is best suited for complex datasets where attention can significantly enhance performance.
 
App Deployment
---------------
The Audio Emotion Recognition Application is a web-based tool designed to analyze audio recordings and detect the emotions expressed within them. Utilizing advanced machine learning models trained on various datasets, this application provides users with insights into the emotional content of audio files.

Features
--------

-   Emotion Detection: Predicts emotions from uploaded audio files and returns probabilities for each emotion.
-   Speech Transcription: Converts spoken language in audio files to text, providing a textual representation of the audio content.
-   Emotional Content Analysis: Interprets the emotional context based on predictions and transcriptions, offering insights into the audio's emotional landscape.
-   User-Friendly Interface: A simple web interface for easy interaction, allowing users to upload audio files and view results effortlessly.

Technologies Used
-----------------

-   Backend Framework: Flask (Python)
-   Audio Processing: librosa
-   Machine Learning: TensorFlow (with ONNX support)
-   Frontend: HTML, CSS, JavaScript

Usage
-----

1.  Upload an Audio File: Use the interface to upload an audio file in supported formats (e.g., WAV, MP3).
2.  View Results: After processing, the application will display the predicted emotions, transcription of the audio, and an interpretation of the emotional content.

### API Endpoints

-   GET /: Renders the main page where users can upload audio files.
-   POST /process_audio: Accepts audio files and returns a JSON response containing:
    -   `Emotion Probabilities`: A breakdown of predicted emotions with their respective probabilities.
    -   `Transcription`: The transcribed text from the audio.
    -   `LLM Interpretation`: Insights into the emotional content based on predictions and transcription.

Conclusion
----------

The Audio Emotion Recognition Application provides a powerful and intuitive tool for analyzing audio recordings. By leveraging machine learning models and advanced audio processing techniques, users can gain valuable insights into the emotions conveyed in speech.
