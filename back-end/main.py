from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Any, Dict, List
import spacy
from src.schemas import TextRequest, PredictionResponse
from src.model import load_models, predict_text
from fastapi.middleware.cors import CORSMiddleware
import os

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variáveis globais que serão carregadas durante o startup
tfidf_vectorizer = Any
classifier_model = Any
CLASS_NAMES = Any  # se você tiver nomes de classes, preencha aqui
# nlp = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """
    Código executado ao iniciar e finalizar a aplicação.
    Carrega os modelos uma única vez.
    """
    global tfidf_vectorizer, classifier_model
    logger.info("Carregando modelos...")
    global nlp
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'pt_core_news_sm-3.7.0-py3-none-any')
    nlp = spacy.load(model_path)
    try:
        tfidf_vectorizer, classifier_model = load_models()

        CLASS_NAMES: List[str] = [
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
        logger.info("Modelos carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelos: {e}")
        # Em produção, talvez queira que a aplicação não inicie se os modelos não carregarem
        raise RuntimeError("Falha ao carregar modelos") from e
    yield
    logger.info("Finalizando aplicação.")


app = FastAPI(
    title="API de Classificação de Texto",
    description="Recebe um texto e retorna a classe prevista e probabilidades.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://classificador-noticias-front.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return Response(status_code=200)

@app.get("/")
async def root() -> Dict[str, str]:
    """Endpoint de saudação."""
    return {"message": "API de Classificação de Texto. Use POST /predict"}


@app.post("/classify", response_model=PredictionResponse)
async def predict(request: TextRequest) -> PredictionResponse:
    """
    Endpoint para classificar um texto.

    - **text**: string com o texto a ser classificado.
    """
    global tfidf_vectorizer, classifier_model, CLASS_NAMES

    # Verifica se os modelos foram carregados
    if tfidf_vectorizer is None or classifier_model is None:
        logger.error("Modelos não carregados.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos não disponíveis no momento. Tente novamente mais tarde.",
        )
    text = request.text
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    text = ' '.join(tokens)

    try:
        predicted_class, probabilities, _ = predict_text(
            text=text,
            tfidf_vectorizer=tfidf_vectorizer,
            classifier=classifier_model,
            class_names=CLASS_NAMES,
        )
    except Exception as e:
        logger.exception("Erro durante a predição")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a classificação: {str(e)}",
        )

    return PredictionResponse(
        predicted_class=predicted_class,
        probabilities=probabilities,
        class_names=[
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
        ],
    )
