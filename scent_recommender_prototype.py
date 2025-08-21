#Scent Recommender System Prototype
#Lauren Denomme, CSU-G Graduate Student in AI/ML, last updated: 5/8/25

#Import relevant libraries
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from tensorflow import keras
import requests
import random
import json

#Import preprocessing libraries
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array

#Import libraries for Keras encoder-decoder model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

#Import library for web GUI
import gradio as gr


#---Step 1: Load the synthetic dataset with example input queries from customers mapped to corresponding scents---
with open("dedcool_scent_data.json", "r") as f:
    synthetic_data = json.load(f)

#Separate input text and target scents
input_texts = [item['input_text'] for item in synthetic_data]
target_scents = ["\t" + item['target_scent'] + "\n" for item in synthetic_data] # Add start and end tokens

#---Step 2: Create Character Vocabularies---
input_tokenizer = Tokenizer(char_level=True)
input_tokenizer.fit_on_texts(input_texts)
input_max_len = max(len(text) for text in input_texts)
input_vocab_size = len(input_tokenizer.word_index) + 1
# Output vocabulary
output_tokenizer = Tokenizer(char_level=True)
output_tokenizer.fit_on_texts(target_scents)
output_vocab_size = len(output_tokenizer.word_index) + 1
output_max_len = max(len(scent) for scent in target_scents)
print(f"Output vocabulary size: {output_vocab_size}")
print(f"Max output sequence length: {output_max_len}")
print("Output Tokenizer Word Index:", output_tokenizer.word_index)

# Convert text to sequences of integers
encoder_input_data = input_tokenizer.texts_to_sequences(input_texts)
decoder_target_data = output_tokenizer.texts_to_sequences(target_scents)

# Pad sequences
encoder_input_padded = pad_sequences(encoder_input_data, maxlen=input_max_len, padding='post')
decoder_target_padded = pad_sequences(decoder_target_data, maxlen=output_max_len, padding='post')

# Prepare decoder input data (shifted target sequences)
decoder_input_padded = pad_sequences([[output_tokenizer.word_index.get('\t', 0)] + seq[:-1] for seq in decoder_target_padded.tolist()], maxlen=output_max_len, padding='post') # Use index of '\t' as start

#One-hot encode sequences
def one_hot_encode(sequences, vocab_size, max_len):
    one_hot = np.zeros((len(sequences), max_len, vocab_size), dtype='float32')
    for i, seq in enumerate(sequences):
        for t, index in enumerate(seq):
            if index > 0:
                one_hot[i, t, index] = 1.
    return one_hot

encoder_input_one_hot = one_hot_encode(encoder_input_padded, input_vocab_size, input_max_len)
decoder_input_one_hot = one_hot_encode(decoder_input_padded, output_vocab_size, output_max_len)

# One-hot encode the target (note: this might need a slight adjustment based on how your loss is calculated)
decoder_target_one_hot = np.zeros((len(decoder_target_padded), output_max_len, output_vocab_size), dtype='float32')
for i, seq in enumerate(decoder_target_padded):
    for t, index in enumerate(seq):
        if index > 0:
            decoder_target_one_hot[i, t, index] = 1

print("\nShape of encoder_input_one_hot:", encoder_input_one_hot.shape)
print("Shape of decoder_input_one_hot:", decoder_input_one_hot.shape)
print("Shape of decoder_target_one_hot:", decoder_target_one_hot.shape)

#---Step 3: Encoder-Decoder Recurrent Neural Network Define Model Function---
# returns train, inference_encoder and inference_decoder models

def define_models(n_input, n_output, n_units):
    '''Defines the training encoder, training decoder, inference encoder, inference decoder'''
    #define training encoder
    encoder_inputs = Input(shape=(None, n_input)) #defines the input layer for the encoder. expects sequences of any length (None), n_input = size of input character library
    encoder = LSTM(n_units, return_state=True) #LSTM layer (RNN for long-range dependencies). n_units = number of hidden units (or memory cells) in the layer. return_state = True represents memory of encoder
    encoder_outputs, state_h, state_c = encoder(encoder_inputs) #applies the LSTM layer to the input sequence. 
    encoder_states = [state_h, state_c] #collect the final hidden and cell states. These encoder states will be used to initialize the decoder.

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output)) #defines the input layer for the decoder. n_output is the size of the output character vocabulary (during training, will be shifted target sequences)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True) #LSTM layer for decoder, has same number of units as encoder LSTM. return_sequences=True tells the LSTM to return outputs at all time steps to compare the decoder's predictions at each time step with the target sequence.
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states) #applies the decoder LSTM to the decoder input sequences. initial_state=encoder_states means that the hidden and cell states of the decoder's LSTM are initialized with the final hidden and cell states of the encoder. 
    decoder_dense = Dense(n_output, activation='softmax') #dense (fully connected) layer that follows the LSTM. the number of units in this layer is equal to the size of the output vocabulary. 
    decoder_outputs = decoder_dense(decoder_outputs) #output of the LSTM is passed through a dense layer to get the final predictions
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs) #defines the complete training model. two inputs as params: the encoder input sequence and the decoder input sequence (shifted target)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states) #creates a separate model that only consists of the encoder

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,)) #defines input layer for the initial hidden state
    decoder_state_input_c = Input(shape=(n_units,)) #defines input layer for the initial cell state
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c] #group the input states
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs) #apply the same decoder LSTM as defined in training. 
    decoder_states = [state_h, state_c] #collect the output states of the decoder LSTM
    decoder_outputs = decoder_dense(decoder_outputs) #apply the same dense layer with softmax activation to get the probability dist. over the next character
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states) #

    # return all models
    return model, encoder_model, decoder_model

n_input = input_vocab_size
n_output = output_vocab_size
n_units = 512  # hyperparam to tune

model, encoder_model, decoder_model = define_models(n_input, n_output, n_units)

#---Step 4: Define and Compile the Model---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# --- Step 5: Train the Model ---
epochs = 500
batch_size = 64

history = model.fit(
    [encoder_input_one_hot, decoder_input_one_hot],
    decoder_target_one_hot,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)
print(history.history)


#---Step 6: Adjusted Predict Sequence Function (output is one-hot) ---
# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, n_features, output_tokenizer):
    # encode
    state = infenc.predict(source, verbose=0)
    # start of sequence input (one-hot encoded for the start character)
    target_seq = np.zeros((1, 1, n_features))
    start_char = '\t' # Assuming '\t' was used as start, if not, we need to find a different way
    start_char_index_output = output_tokenizer.word_index.get(start_char)
    if start_char_index_output is not None:
        target_seq[0, 0, start_char_index_output] = 1.
    elif 1 in output_tokenizer.word_index.values():
        # Fallback: if '\t' not found, use the character with index 1 (if it exists)
        for char, index in output_tokenizer.word_index.items():
            if index == 1:
                target_seq[0, 0, index] = 1.
                break
    else:
        # Last resort: if index 1 not found, maybe the first char in vocab?
        if output_tokenizer.word_index:
            first_char = next(iter(output_tokenizer.word_index))
            first_index = output_tokenizer.word_index[first_char]
            target_seq[0, 0, first_index] = 1.

    # collect predictions (one-hot encoded)
    output = []
    for _ in range(n_steps):
        yhat, h, c = infdec.predict([target_seq] + state, verbose=0)
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence (using argmax to get the most likely char index)
        predicted_index = np.argmax(yhat[0, 0, :])
        new_target_seq = np.zeros((1, 1, n_features))
        new_target_seq[0, 0, predicted_index] = 1.
        target_seq = new_target_seq
    return np.array(output)

# --- Step 7 & 8: Prediction and Decoding (with adjusted functions) ---
def decode_one_hot_sequence(one_hot_seq, tokenizer):
    reverse_word_index = {i: char for char, i in tokenizer.word_index.items()}
    decoded_chars = []
    for vector in one_hot_seq:
        sampled_token_index = np.argmax(vector)
        sampled_char = reverse_word_index.get(sampled_token_index)
        if sampled_char is None:
            break
        decoded_chars.append(sampled_char)
    return ''.join(decoded_chars)

input_sequence_index = 0

#Get the input sequence
input_seq = encoder_input_padded[input_sequence_index]
#Decode the input sequence to text
decoded_input = ''.join([input_tokenizer.index_word.get(i, '') for i in input_seq if i > 0])
print("Input Text:", decoded_input)

input_text_example = input_texts[input_sequence_index]
input_sequence_pred = input_tokenizer.texts_to_sequences([input_text_example])
padded_input_pred = pad_sequences(input_sequence_pred, maxlen=input_max_len, padding='post')
encoder_input_pred_one_hot = one_hot_encode(padded_input_pred, input_vocab_size, input_max_len)

predicted_one_hot = predict_sequence(encoder_model, decoder_model, encoder_input_pred_one_hot, output_max_len, output_vocab_size, output_tokenizer)
predicted_scent = decode_one_hot_sequence(predicted_one_hot, output_tokenizer)

def recommend_scent(user_input):
    # Preprocess the user input
    input_sequence = input_tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(input_sequence, maxlen=input_max_len, padding='post')
    encoder_input = one_hot_encode(padded_input, input_vocab_size, input_max_len)

    # Get the one-hot encoded prediction
    predicted_one_hot = predict_sequence(encoder_model, decoder_model, encoder_input, output_max_len, output_vocab_size, output_tokenizer)

    # Decode the one-hot prediction to a scent string
    predicted_scent = decode_one_hot_sequence(predicted_one_hot, output_tokenizer)

    # Remove potential start and end tokens from the output
    predicted_scent = predicted_scent.replace('\t', '').replace('\n', '').strip()
    return predicted_scent

iface = gr.Interface(
    fn=recommend_scent,
    inputs=gr.Textbox(label="Describe the scent you're looking for:"),
    outputs=gr.Textbox(label="Recommended Scent:")
)

iface.launch(share=False) # Set share=True to create a public link
