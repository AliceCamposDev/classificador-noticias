# Classificador de Notícias - (Back-end)

Este projeto é o backend de um classificador de notícias em português. Ele utiliza um modelo de Machine Learning treinado com TF‑IDF e Regressão Logística para categorizar textos jornalísticos em 24 classes predefinidas, baseadas em editorias dos sites Folha e UOL.

A API foi desenvolvida com **FastAPI** e está disponível em produção no endereço:  
[https://classificador-noticias.vercel.app/](https://classificador-noticias.vercel.app/)

---

## 📚 Sobre o modelo

- **Dados**: [News of the site Folha/UOL](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol) (Kaggle)
- **Pré‑processamento**: concatenação do título com o texto, remoção de stopwords e lematização com spaCy (modelo `pt_core_news_sm`).
- **Feature extraction**: TF‑IDF
- **Classificador**: Regressão Logística
- **Acurácia**: 85,95%
- **Classes**: 24 categorias (ex.: `ambiente`, `ciencia`, `colunas`, `comida`, `cotidiano`, `educacao`, …)

---

## 🛠️ Tecnologias

- Python 3.10+
- FastAPI
- spaCy
- scikit‑learn
- Uvicorn
- Docker (opcional)

---

## 🚀 Como executar localmente

### 🔧 Pré‑requisitos

- Python 3.10 ou superior
- pip
- (Opcional) Docker

1. Clone o repositório

git clone https://github.com/AliceCamposDev/classificador-noticias
cd classificador-noticias/back-end

2. Instale as dependências

pip install -r requirements.txt

3. Execute a aplicação

python -m uvicorn main:app --reload

🐳 Executando com Docker

sudo docker compose up --build

---

## 📬 Endpoints da API
GET /
Descrição: Mensagem de boas‑vindas.

Resposta:

json
{
  "message": "API de Classificação de Texto. Use POST /predict para enviar seu texto\n/classnames para ver as possiveis respostas\n/docs para o swagger "
}

POST /classify
Descrição: Recebe um texto e retorna a classe prevista e as probabilidades para todas as classes.

Body (JSON):

json
{
  "text": "O governo anunciou novas medidas econômicas para controle da inflação...."
}
Resposta (exemplo):

json
{
  "predicted_class": "mercado",
  "probabilities": [0.05, 0.1, 0.3, 0.01, ...],
  "class_names": ["ambiente", "ciencia", ...]
}

GET /classnames
Descrição: Retorna a lista de todas as classes possíveis.

Resposta:

json
["ambiente", "ciencia", "colunas", ...]
GET /docs e GET /redoc
Documentação interativa gerada automaticamente pelo FastAPI.
