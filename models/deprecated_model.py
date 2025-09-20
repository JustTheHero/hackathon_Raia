import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FakeNewsWOW:
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=500):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"shape: {df.shape}")
        print(f"distribuicao:\n{df['label'].value_counts()}")
        print(f"NUll valores:\n{df.isnull().sum()}")
        
        texts = df['preprocessed_news'].astype(str).tolist()
        labels = (df['label'] == 'fake').astype(int).tolist()  
        
        return texts, labels
    
    def prepare_sequences(self, texts, labels):
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        X = pad_sequences(sequences, maxlen=self.max_length, truncating='post')
        y = np.array(labels)
        
        print(f"shape: {X.shape}")
        
        return X, y
    
    def build_model(self):
        self.model = Sequential()
        
        self.model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length
        ))
        
        
        self.model.add(Conv1D(128, 5, activation='relu'))
        self.model.add(GlobalMaxPooling1D())
        
        
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, X, y, validation_split=0.2, epochs=10, batch_size=32):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, X_val, y_val
    
    def evaluate_model(self, X_val, y_val):
        y_pred_prob = self.model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Real', 'Fake']))
        
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return y_pred, y_pred_prob
    
    def predict_text(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length)
        
        prediction = self.model.predict(padded)[0][0]
        label = "Fake" if prediction > 0.5 else "Real"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return label, confidence

def main():
    classifier = FakeNewsWOW(vocab_size=10000, embedding_dim=128, max_length=300)
    
    texts, labels = classifier.load_data('../Fake.br-Corpus/preprocessed/pre-processed.csv')
    
    X, y = classifier.prepare_sequences(texts, labels)
    
    classifier.build_model()
    print(classifier.model.summary())
    
    history, X_val, y_val = classifier.train_model(X, y, epochs=15)
    
    y_pred, y_pred_prob = classifier.evaluate_model(X_val, y_val)
    
    classifier.plot_training_history(history)
    
    val_accuracy = max(history.history['val_accuracy'])
    
    print(f"\nAccuracy: {val_accuracy:.4f}")
    
    sample_text = "lula morte prisao recebeu justiça investigacao"
    prediction, confidence = classifier.predict_text(sample_text)
    print(f"\nSample prediction:")
    print(f"Text: {sample_text}")
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
