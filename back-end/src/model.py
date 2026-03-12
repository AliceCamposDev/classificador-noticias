import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_models() -> Tuple[object, object]:
    """
    Carrega o vetorizador TF-IDF e o modelo de regressão logística.
    Retorna uma tupla (vetorizador, modelo).
    """
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    model_path = MODELS_DIR / "logistic_model.pkl"
    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)
    return tfidf, model

def predict_text(
    text: str,
    tfidf_vectorizer: Any,
    classifier: Any,
    class_names: Optional[List[str]] = None
) -> Tuple[int, List[float], Optional[List[str]]]:
    """
    Recebe um texto e retorna a classe prevista e as probabilidades.
    """
    # O vetorizador espera uma lista de strings (iterável)
    X = tfidf_vectorizer.transform([text])

    pred: np.ndarray = classifier.predict(X)           # array de inteiros
    proba: np.ndarray = classifier.predict_proba(X)    # array 2D

    predicted_class = int(pred[0])
    probabilities = proba[0].tolist()                  # converte para lista Python

    return predicted_class, probabilities, class_names