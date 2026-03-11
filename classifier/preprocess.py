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

# ===================== FUNÇÕES DE PROCESSAMENTO (SÃO DEFINIDAS NO ESCOPO PRINCIPAL) =====================

def load_data(filepath: str = "processed_data.csv") -> pd.DataFrame:
    """Carrega os dados tratados a partir de um arquivo CSV."""
    if not os.path.exists(filepath):
        print("Dados não puderam ser carregados")
        raise FileNotFoundError(f"Arquivo {filepath} não encontrado.")
    df = pd.read_csv(filepath)
    return df

def concat_txt(df: pd.DataFrame) -> pd.DataFrame:
    """Concatena título e texto em uma nova coluna 'texto_completo'."""
    df['texto_completo'] = df['title'] + ' ' + df['text']
    return df

def gen_report(
    df: pd.DataFrame,
    output_file: str = "report.txt",
    save: bool = True,
    print_report: bool = False,
) -> None:
    """Gera relatório com informações do DataFrame."""
    str_analysis: str = (
        f"Dataset shape: {df.shape}\nColumns: {list(df.columns)}\n\nMissing Values:\n{df.isnull().sum()}\n\n"
        f"Numeric Summary:\n{df.describe(include='all')}\n\nFirst 5 rows:\n{df.head().to_string()}\n"
    )
    if print_report:
        print(str_analysis)
    if save:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str_analysis)
        print(f"\nRelatório salvo em {output_file}")

# ===================== FUNÇÕES PARA PROCESSAMENTO EM PARALELO =====================

def init_worker() -> None:
    global nlp
    nlp = spacy.load('pt_core_news_sm')

def process_chunk(texts_chunk: List[str]) -> List[str]:
    """
    Processa um lote de textos: lematiza e remove stopwords/pontuação.
    Utiliza o modelo spaCy carregado globalmente no worker.
    """
    global nlp
    results = []
    for text in texts_chunk:
        # Processa o texto com spaCy
        doc = nlp(text)
        # Extrai os lemas, ignorando stopwords e pontuação
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        # Junta os tokens em uma string novamente
        results.append(' '.join(tokens))
    return results

def process_in_parallel(
    df: pd.DataFrame,
    text_column: str,
    num_processes: Optional[int] = None
) -> List[str]:
    """
    Aplica lematização e remoção de stopwords em paralelo a uma coluna de texto.
    Retorna uma lista com os textos processados na mesma ordem do DataFrame.
    """
    if num_processes is None:
        num_processes = 10
    # Garantir que não criamos mais processos que o necessário
    num_processes = min(num_processes, len(df))

    # Extrai os textos da coluna especificada
    texts = df[text_column].tolist()
    # Divide em chunks para cada processo
    chunk_size = len(texts) // 5000
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # Cria o pool de processos com o inicializador que carrega o modelo
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        # Mapeia os chunks para processamento paralelo
        chunk_results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing chunks"))
        
    # Achata a lista de listas
    processed_texts = [item for sublist in chunk_results for item in sublist]
    return processed_texts

if __name__ == "__main__":
    print("Carregando dados...")
    df = load_data()
    cols_to_drop= ["text", "texto_completo"]
    
    existing_cols: List[str] = [col for col in cols_to_drop if col in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"Colunas removidas: {existing_cols}")
    else:
        print("Colunas não encontrada para remoção.")
    # print("Concatenando título e texto...")
    # df = concat_txt(df)

    # print("Processando textos em paralelo...")
    # df['texto_processado'] = process_in_parallel(df, 'texto_completo', num_processes=10)

    # print("Gerando relatório...")
    # gen_report(df, print_report=True)

    df.to_csv("processed_data_crop.csv", index=False)
    # print("Arquivo 'processed_data.csv' salvo.")