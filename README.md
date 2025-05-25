# 🧠 Next Word Predictor using LSTM

This project implements a **Next Word Predictor** using a Long Short-Term Memory (LSTM) neural network. It learns from a given text corpus and generates the next most probable word(s) based on user input. The model is developed using TensorFlow and demonstrates essential Natural Language Processing (NLP) techniques such as tokenization, sequence padding, and text generation.

---

## 📌 Features

- Trains an LSTM-based language model on a text corpus
- Predicts the next word(s) based on a seed sentence
- Uses word-level tokenization with padded input sequences
- Simple and extendable design with TensorFlow/Keras

---

## 🧠 How It Works

1. **Text Preprocessing**
   - Reads and tokenizes text
   - Converts text to sequences using a word tokenizer
   - Pads input sequences for uniform shape

2. **Model Architecture**
   - Embedding layer (100 dimensions)
   - LSTM layer (150 units)
   - Dense layer with softmax activation for word prediction

3. **Training**
   - Uses categorical cross-entropy loss
   - Trained over 10 epochs on the processed dataset

4. **Prediction**
   - Predicts the next word(s) based on a user-provided seed text

---

## ✅ Example Usage

```python
predict_next_word("sherlock holmes was", 3)
# Output: "sherlock holmes was not in"

