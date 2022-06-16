from cmath import e
from matplotlib.font_manager import json_load
from flask import Flask, request, jsonify
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('model.h5')
app = Flask(__name__)

def get_padded_value(entry):
    vocabs = json.load(open('vocabs.txt'))
    entry_arr = entry.split(' ')
    entry_sequence = []
    for i in range(0, 200):
        try:
            entry_sequence.append(vocabs[entry_arr[i]])
        except KeyError:
            entry_sequence.append(1)
        except IndexError:
            entry_sequence.append(0)
    entry_sequence = [entry_sequence]
    entry_padded = pad_sequences(entry_sequence, maxlen=200, padding='post', truncating='post')
    return entry_padded

@app.route('/')
def index():
    return "Hello World!"

@app.route('/predict', methods = ["POST"])
def predict():
    entry = request.form.get('entry')
    entry_padded = get_padded_value(entry)
    prediction = model.predict(entry_padded)
    labels = ['neutral', 'nsfw', 'suicide', 'cyberbullying']
    return jsonify({'prediction':str(labels[np.argmax(prediction)])})

if __name__ == '__main__':
    app.run(debug=True)