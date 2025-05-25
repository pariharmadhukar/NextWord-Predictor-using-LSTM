import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from nltk.tokenize import word_tokenize
import nltk


# Load and preprocess text
with open("1661-0.txt", "r", encoding='utf-8') as file:
    text = file.read()

# Optional: Clean text (remove special chars, lowercase etc.)
text = text.lower()
tokens = word_tokenize(text)

# Prepare sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for i in range(1, len(tokens)):
    n_gram_sequence = tokens[:i+1]
    encoded = tokenizer.texts_to_sequences([' '.join(n_gram_sequence)])[0]
    if len(encoded) > 1:
        input_sequences.append(encoded)

# Pad sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

# Features and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_seq_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X, y, epochs=10, verbose=1)

def predict_next_word(seed_text, num_words=1):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = tokenizer.index_word[np.argmax(predicted)]
        seed_text += ' ' + predicted_word
    return seed_text

# Example usage
print(predict_next_word("sherlock holmes was", 3))
