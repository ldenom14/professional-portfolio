# Scent Recommender System: A Character-Level Encoder-Decoder Model
This project is a prototype of a scent recommendation system built using a character-level encoder-decoder recurrent neural network (RNN) with TensorFlow and Keras. 
The model takes a user's free-text description of a desired scent (e.g., "I want a warm, cozy scent with hints of vanilla") and recommends a specific scent name. 
This capstone project showcases my ability to develop innovative, end-to-end AI/ML solutions from data preparation to deployment.

# Key Features and Technologies
+ **Custom Encoder-Decoder Architecture:** The core of the system is a sequence-to-sequence model built with LSTM layers. The encoder processes the user's descriptive input and compresses it into a "thought vector,"
which the decoder then uses to generate the output scent name, character by character.

+ **End-to-End Implementation:** The project demonstrates the full machine learning lifecycle, from building and preprocessing a synthetic dataset to training the model and developing an interactive web interface.

+ **Full Data Preprocessing Pipeline:** I implemented key data preparation steps from scratch, including character-level tokenization, one-hot encoding, and sequence padding, to prepare unstructured text data for the neural network.

+ **Interactive Web GUI:** The model is deployed with a Gradio interface, providing a user-friendly and functional demonstration of the system's capabilities.

**Technologies Used:** Python, TensorFlow, Keras, scikit-learn, Gradio

# How It Works
The system operates in two main phases:

**1. The Encoder**
The encoder-LSTM reads the user's input text one character at a time. It learns to understand the meaning and context of the input, condensing all the information into a single vector of numbers, known as the "context vector" or "thought vector."

**2. The Decoder**
The decoder-LSTM is initialized with the encoder's context vector. It uses this vector to predict the target scent name one character at a time. At each step, it predicts the next most likely character based on its internal state and the characters it has already generated, until it predicts an end-of-sequence token.

# How to Run the Prototype
To run this prototype on your local machine:

+ Clone this repository.

+ Install the required libraries: pip install tensorflow gradio

+ Ensure the dedcool_scent_data.json file is in the same directory.

+ Run the script: python scent_recommender_prototype.py

This will launch the Gradio web interface in your browser, where you can input a scent description and receive a recommendation.






