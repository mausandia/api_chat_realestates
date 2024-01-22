import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import random

nltk.download('punkt')
nltk.download('wordnet')

# Load the intents file
with open('intents.json') as file:
    intents = json.load(file)

# Extract data from intents
all_words = []
tags = []
xy = []

lemmatizer = WordNetLemmatizer()

# Preprocess the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = word_tokenize(pattern)
        all_words.extend(words)
        xy.append((words, intent['tag']))
    tags.append(intent['tag'])

# Lemmatize and lowercase all words
all_words = [lemmatizer.lemmatize(w.lower()) for w in all_words if w.isalnum()]

# Remove duplicates and sort words
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

# Create training data
X_train = []
y_train = []

for (pattern_words, tag) in xy:
    bag = [1 if lemmatizer.lemmatize(word.lower()) in pattern_words else 0 for word in all_words]
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(tags), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Save the preprocessing data
import pickle

with open('chatbot_data.pickle', 'wb') as f:
    pickle.dump({'words': all_words, 'tags': tags, 'xy': xy}, f)

print(f"Chat model saved!")
