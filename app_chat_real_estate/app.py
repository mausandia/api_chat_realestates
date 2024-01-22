from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json

app = Flask(__name__)

# Load the preprocessed data
with open('app_chat_real_estate/chatbot_data.pickle', 'rb') as f:
    data = pickle.load(f)
    words = data['words']
    tags = data['tags']

# Load the trained model
model = tf.keras.models.load_model('app_chat_real_estate/chatbot_model.h5')

# Lemmatizer for tokenization
lemmatizer = WordNetLemmatizer()

def process_user_input(user_input):
    # Tokenize and lemmatize user input
    input_words = word_tokenize(user_input)
    input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words if word.isalnum()]

    # Convert input to bag of words
    input_bag = [1 if lemmatizer.lemmatize(word.lower()) in input_words else 0 for word in words]

    # Make prediction
    predictions = model.predict(np.array([input_bag]))
    predicted_tag = tags[np.argmax(predictions)]

    # Find the intent
    with open('app_chat_real_estate/intents.json') as file:
        intents = json.load(file)
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return intent

# API endpoint for user input
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['user_input']
        intent = process_user_input(user_input)
        response = random.choice(intent['responses'])

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
