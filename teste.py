import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import nltk
nltk.download('rslp')

try:
    df = pd.read_csv('pre-processed.csv', index_col='index')
except FileNotFoundError:
    print("não abriu")
    sys.exit(1)

if df['label'].dtype == 'object':
    df['label_numeric'] = df['label'].map({'fake': 1, 'true': 0})
else:
    df['label_numeric'] = df['label']

if df['label_numeric'].isnull().any():
    df = df.dropna(subset=['label_numeric'])

if df['preprocessed_news'].isnull().any():
    df = df.dropna(subset=['preprocessed_news'])

X_text = df['preprocessed_news']
y = df['label_numeric'].astype(int)

print(f"Tamanho dataset{len(y)} ")
print(f"Contagem de classes: \n{y.value_counts()}")

vectorizer = CountVectorizer(binary=True)
X_vec = vectorizer.fit_transform(X_text)

print(f"Matriz de features (X) criada com shape: {X_vec.shape}")

n_samples = len(y)
n_classes = len(y.unique())

if n_samples < 2 or n_classes < 2:
    min_class_count = y.value_counts().min()
    cv_folds = min(5, min_class_count)

    if cv_folds < 2:

        if cv_folds < 5:
            use_dual = X_vec.shape[1] < X_vec.shape[0]
            clf = LinearSVC(dual=use_dual, max_iter=2000, C=1.0)

            try:
                y_pred = cross_val_predict(clf, X_vec, y, cv=cv_folds)
            except ValueError as e:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                y_pred = cross_val_predict(clf, X_vec, y, cv=kf)

            target_names = ['verdadeiro (0)', 'fake (1)']
            print(classification_report(y, y_pred, target_names=target_names, zero_division=0))

            print("\nMatriz de Confusão")
            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            print(f"     [V (0)]  [F (1)]")
            print(f"[V(0)] {cm[0][0]:>5}   {cm[0][1]:>5}")
            print(f"[F(1)] {cm[1][0]:>5}   {cm[1][1]:>5}")
