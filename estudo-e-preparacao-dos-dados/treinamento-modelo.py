import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from multiprocessing import Pool, cpu_count
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def load_data(filepath: str = "processed_data_crop.csv") -> pd.DataFrame:
    """Carrega os dados tratados a partir de um arquivo CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo {filepath} não encontrado.")
    df = pd.read_csv(filepath)
    return df

if __name__ == "__main__":
    print("Carregando dados...") 
    df = load_data()

    # Codificar categorias
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    # Verificar distribuição das classes
    class_counts = df['category_encoded'].value_counts().sort_index()
    print("Distribuição original das classes (encoded):")
    
    # Filtrar classes com poucas amostras
    min_samples = 200  # ajuste conforme necessário
    classes_to_keep = class_counts[class_counts >= min_samples].index
    # asmais
    df_filtered = df[df['category_encoded'].isin(classes_to_keep)].copy()
    low_count_indices  = class_counts[class_counts <= min_samples].index
    low_count_names = le.inverse_transform(low_count_indices)
    print(le.inverse_transform(classes_to_keep))

    # Re-ajustar o encoder para as classes mantidas (opcional, mas evita índices esparsos)
    le_filtered = LabelEncoder()
    df_filtered['category_encoded'] = le_filtered.fit_transform(df_filtered['category'])
    
    print(f"\nApós filtrar classes com < {min_samples} amostras:")
    print(f"Shape original: {df.shape}")
    print(f"Shape filtrado: {df_filtered.shape}")
    print(f"Classes mantidas: {len(le_filtered.classes_)}")
    
    # Separar features e target
    X = df_filtered['texto_processado']
    y = df_filtered['category_encoded']
    
    print("Dividindo treino/validação/teste...")
    # Divisão treino (80%) / temporário (20%)
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Dividir temporário em validação (50% de 20% = 10%) e teste (50% de 20% = 10%)
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Tamanhos: treino={len(X_train_text)}, val={len(X_val_text)}, teste={len(X_test_text)}")
    
    # ========== MODELO COM TF-IDF ==========
    print("\nCriando features TF-IDF...")
    tfidf = TfidfVectorizer(max_features=7000000, ngram_range=(1,3), min_df=30, max_df=0.7)
    with tqdm(total=3, desc="TF-IDF steps") as pbar:
        X_train_tfidf = tfidf.fit_transform(X_train_text)
        pbar.update(1)
        X_val_tfidf = tfidf.transform(X_val_text)
        pbar.update(1)
        X_test_tfidf = tfidf.transform(X_test_text)
        pbar.update(1)
    
    print("Treinando Regressão Logística com TF-IDF...")
    clf_tfidf = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42, verbose=1) #newton-cg lbfgs
    clf_tfidf.fit(X_train_tfidf, y_train)
    
    y_val_pred_tfidf = clf_tfidf.predict(X_val_tfidf)
    val_acc_tfidf = accuracy_score(y_val, y_val_pred_tfidf)
    print(f"Acurácia na validação (TF-IDF): {val_acc_tfidf:.4f}")
    print("\nRelatório de Classificação - Validação (TF-IDF):")
    print(classification_report(y_val, y_val_pred_tfidf, target_names=le_filtered.classes_))
    
    # Teste
    y_test_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
    test_acc_tfidf = accuracy_score(y_test, y_test_pred_tfidf)
    print(f"Acurácia no teste (TF-IDF): {test_acc_tfidf:.4f}")
    import joblib

# Salvar
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    joblib.dump(clf_tfidf, 'logistic_model.pkl')