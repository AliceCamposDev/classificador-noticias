from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Any, Dict, List, AsyncIterator
import spacy
from src.schemas import TextRequest, PredictionResponse
from src.model import load_models, predict_text
from fastapi.middleware.cors import CORSMiddleware
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global vars, não gosto de usar mas é um jeito simples de resolver o problema.
tfidf_vectorizer: Any = None
classifier_model: Any = None
nlp: Any = None
class_names: List[str] = [
    "ambiente",
    "bbc",
    "ciencia",
    "colunas",
    "comida",
    "cotidiano",
    "educacao",
    "empreendedorsocial",
    "equilibrioesaude",
    "esporte",
    "folhinha",
    "ilustrada",
    "ilustrissima",
    "mercado",
    "mundo",
    "opiniao",
    "paineldoleitor",
    "poder",
    "saopaulo",
    "seminariosfolha",
    "sobretudo",
    "tec",
    "turismo",
    "tv",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Carrega os modelos usados ao iniciar
    ps: o modelo pt_core_news_sm esta mockado no codigo porque a vercel não consegue carregar o modelo pelo tamanho direito
    
    Args:
        app (FastAPI):Instancia da aplicação


    Returns:
        RuntimeError: Caso ocorra falha ao carregar os recursos
        necessários durante o startup.

    Raises:
        RuntimeError: Modelos não foram carre
    """
    global tfidf_vectorizer, classifier_model, nlp
    logger.info("Carregando modelos...")
    try:
        with tqdm(total=3, desc="Loading model") as pbar:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                "pt_core_news_sm",
                "pt_core_news_sm",
                "pt_core_news_sm-3.7.0",
            )
            pbar.update(1)
            nlp = spacy.load(model_path)
            pbar.update(1)
            tfidf_vectorizer, classifier_model = load_models()
            pbar.update(1)
        logger.info("Modelos carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        raise RuntimeError("Falha ao carregar modelos") from e
    yield
    logger.info("Finalizando aplicação.")


app = FastAPI(
    title="API de Classificação de Texto",
    description="Recebe um texto e retorna a classe prevista e probabilidades.",
    version="1.0.1",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://classificador-noticias-front.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Rota raiz, apenas retorna uma mensagem

    Returns:
        Dict[str, str]: mensagem
    """
    return {
        "message": "API de Classificação de Texto. Use POST /predict para enviar seu texto"
    }


@app.post("/classify", response_model=PredictionResponse)
async def predict(request: TextRequest) -> PredictionResponse:
    """Função que classifica o texto

    pre processa o texto, lematizando e tirando as keywords e aplica o modelo treinado para classificar o texto a partir de tfidf

    Args:
        request (TextRequest): request com o texto a ser analisado

    Raises:
        HTTPException: 503 service unavilable - Modelos não foram carregados
        HTTPException: 500 internal server error - Erro interno durante  o processamento do texto
        HTTPException: 500 internal server error - Erro interno durante  a classificação do texto

    Returns:
        PredictionResponse: {
            predicted_class: classificação predita
            probabilities: probabilidade de todas as classificações possiveis
            class_names: nome das classes possiveis
        }
    """
    global class_names

    if tfidf_vectorizer is None or classifier_model is None:
        logger.error("Modelos não carregados.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos não disponíveis no momento. Tente novamente mais tarde.",
        )
    try:
        text = request.text
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]
        text = " ".join(tokens)
        logger.info("texto preprocessado")
    except Exception as e:
        logger.error("Erro interno durante o processamento do texto")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante  o processamento do texto: {str(e)}",
        )
    try:
        predicted_class, probabilities, _ = predict_text(
            text=text, tfidf_vectorizer=tfidf_vectorizer, classifier=classifier_model, class_names=class_names
        )
    except Exception as e:
        logger.exception("Erro durante a predição")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a classificação: {str(e)}",
        )
    # probabilities deveria estar em um dict com os classnames? sim, mas não quero mudar o front-end
    return PredictionResponse(
        predicted_class=predicted_class,
        probabilities=probabilities,
        class_names=class_names,
    )

@app.get("/classnames")
async def get_classnames() -> list[str]:
    """Função para pegar as classificações possíveis

    Returns:
        List[str]: classnames
    """
    return class_names
    
