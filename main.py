import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
LSTM_DIM = 64
DENSE_DIM = 64
TRUNCATING_TYPE = PADDING_TYPE = 'post'
OOV_TOK = '<OOV>'

sentences = []
labels = []

df = pd.read_csv('DATASET.csv', names=['label', 'text'])
df = df.sample(frac=1).reset_index(drop=True)

for index, row in df.iterrows():
  if row['label'] == "depression" or 'http' in str(row['text']): continue

  labels.append(row['label'])
  article = str(row['text'])

  for word in STOPWORDS:
    token = ' ' + word + ' '
    article = article.replace(token, ' ')
    article = article.replace(' ', ' ')

  sentences.append(article)

train_size = int(len(sentences) * .8)

training_sentences = sentences[0: train_size]
training_labels = labels[0: train_size]

testing_sentences = sentences[train_size:]
testing_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token=OOV_TOK, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{"}~\t\n')
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

print(set(labels))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_labels_seq = np.array(label_tokenizer.texts_to_sequences(training_labels))
testing_labels_seq = np.array(label_tokenizer.texts_to_sequences(testing_labels))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LENGTH),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 10

history = model.fit(training_padded, training_labels_seq, epochs=num_epochs, validation_data=(testing_padded, testing_labels_seq), verbose=1)
test_loss, test_acc = model.evaluate(testing_padded, testing_labels_seq)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False

# tflite_model = converter.convert()

# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

sample_text =["nigga"]
sample_text_seq = tokenizer.texts_to_sequences(sample_text)
padded_sample_text = pad_sequences(sample_text_seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
print(padded_sample_text)
prediction = model.predict(padded_sample_text)
print(np.argmax(prediction))
labels = ['', 'suicide','cyberbullying', 'neutral', 'nsfw']
print(prediction, labels[np.argmax(prediction)])

print(len(tokenizer.word_index))
lol = tokenizer.word_index
for key in lol:
    with open('somefile.txt', 'a', encoding="utf-8") as the_file:
        the_file.write(str(key))
        if not '\n' in str(key):
            the_file.write('\n')

import pickle
pickle.dump(model, open('model.pkl','wb'))

# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")