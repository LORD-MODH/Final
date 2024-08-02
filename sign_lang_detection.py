import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

@st.cache
def load_data(train_url, test_url):
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    return train_df, test_df

def preprocess_data(df):
    X = df.drop('label', axis=1).values
    y = df['label'].values
    X = X / 255.0
    X = X.reshape(-1, 28, 28, 1)
    y = to_categorical(y, num_classes=24)
    return X, y

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(24, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return classification_report(y_true, y_pred_classes), confusion_matrix(y_true, y_pred_classes), f1_score(y_true, y_pred_classes, average='weighted')

def main():
    st.title("Sign Language Detection")

    train_url = 'sign_mnist_train.csv'
    test_url = 'sign_mnist_test.csv'

    train_df, test_df = load_data(train_url, test_url)

    if st.button('Show Data'):
        st.write(train_df.head())

    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = build_model()

    if st.button('Train Model'):
        model = train_model(model, X_train, y_train, X_val, y_val)
        st.write("Model trained successfully!")

    if st.button('Evaluate Model'):
        report, cm, f1 = evaluate_model(model, X_test, y_test)
        st.write("Classification Report:\n", report)
        st.write("Confusion Matrix:\n", cm)
        st.write("F1 Score:", f1)

    if st.button('Upload Image for Prediction'):
        uploaded_file = st.file_uploader("Choose an image...", type="png")
        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file).convert('L').resize((28, 28))).reshape(1, 28, 28, 1) / 255.0
            prediction = np.argmax(model.predict(image), axis=1)
            st.write(f"Predicted Label: {chr(prediction[0] + 65)}")

if __name__ == '__main__':
    main()
