import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

@st.cache
def load_data():
    url = 'https://drive.google.com/uc?id=1ZKVzTnCE-U5uMkopcBsPNj0LFtPTX3z4'
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    tokenizer = Tokenizer(num_words=5000, split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    y = pd.get_dummies(df['label']).values
    return X, y, tokenizer

def build_model(input_length):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=256, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=2)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return classification_report(y_true, y_pred_classes), confusion_matrix(y_true, y_pred_classes), f1_score(y_true, y_pred_classes, average='weighted')

def main():
    st.title("Fake News Classifier")

    df = load_data()

    if st.button('Show Data'):
        st.write(df.head())

    X, y, tokenizer = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = build_model(X.shape[1])

    if st.button('Train Model'):
        model = train_model(model, X_train, y_train, X_val, y_val)
        st.write("Model trained successfully!")

    if st.button('Evaluate Model'):
        report, cm, f1 = evaluate_model(model, X_test, y_test)
        st.write("Classification Report:\n", report)
        st.write("Confusion Matrix:\n", cm)
        st.write("F1 Score:", f1)

    news_text = st.text_area("Enter news text to classify")

    if st.button('Classify'):
        if news_text:
            seq = tokenizer.texts_to_sequences([news_text])
            padded = pad_sequences(seq, maxlen=X.shape[1])
            pred = model.predict(padded)
            label = np.argmax(pred, axis=1)
            st.write("Prediction:", "Real" if label == 1 else "Fake")

if __name__ == '__main__':
    main()
