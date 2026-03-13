import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

def load_models() -> Tuple[object, object]:
    """Carrega os modelos tfidf e vectorizer

    Returns:
        Tuple[object, object]: both models
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
    class_names: Optional[List[str]],
) -> Tuple[int, List[float], Optional[List[str]]]:
    """Função que usa os modelos para classificar os textos

    Args:
        text (str): texto a ser classificado
        tfidf_vectorizer (Any): vetorizer
        classifier (Any): modelo classificador
        class_names (Optional[List[str]], optional): classes possiveis para classificação, nem precisa dele na real, mas ta ai

    Returns:
        Tuple[int, List[float], Optional[List[str]]]: classe predita, probabilidades de todas as classes, nome das classes possiveis
    """    
    X = tfidf_vectorizer.transform([text])

    pred: np.ndarray = classifier.predict(X)
    proba: np.ndarray = classifier.predict_proba(X)

    predicted_class = int(pred[0])
    probabilities = proba[0].tolist()

    return predicted_class, probabilities, class_names