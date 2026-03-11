import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime

plt.style.use("ggplot")
sns.set_style("whitegrid")


def load_and_save_data(
    dataset_slug: str = "marlesson/news-of-the-site-folhauol",
    output_path: str = "folha_news.csv",
) -> pd.DataFrame:
    """Baixa o dataset do Kaggle, localiza o arquivo CSV principal,
    carrega os dados em um DataFrame e os salva em um arquivo local.

    Args:
        dataset_slug (str, optional): Identificador do dataset no Kaggle.
        output_path (str, optional): Caminho onde o arquivo CSV será salvo.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.

    Raises:
        FileNotFoundError: Se nenhum arquivo CSV for encontrado no dataset.
        Exception: Se ocorrer um erro ao carregar o arquivo CSV.
    """
    dataset_path: str = kagglehub.dataset_download(dataset_slug)

    csv_files: list[str] = []
    for root, _dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado no dataset.")

    main_file: str = csv_files[0]
    print(f"Carregando arquivo: {main_file}")

    try:
        df: pd.DataFrame = pd.read_csv(main_file)
    except UnicodeDecodeError:
        df = pd.read_csv(main_file, encoding="latin1")
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        raise

    # salvar dados local
    df.to_csv(output_path, index=False)
    print(f"DataFrame salvo em: {output_path}")

    return df


def add_weekday_column(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    if date_column not in df.columns:
        raise KeyError(f"Coluna '{date_column}' não encontrada no DataFrame.")

    date_series: pd.Series = pd.to_datetime(df[date_column], errors="coerce")

    n_invalid: int = date_series.isna().sum()
    if n_invalid > 0:
        print(
            f"Aviso: {n_invalid} valores não puderam ser convertidos para data e serão NaN na coluna 'weekday'."
        )
    df["weekday"] = date_series.dt.day_name()

    return df


def clean(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """
    Remove as colunas especificadas (se existirem) e
    concatena as colunas 'title' e 'text' em uma nova coluna 'full_text'.

    Args:
        df(pd.DataFrame): DataFrame original contendo as colunas esperadas.
        cols_to_drop: Lista de colunas a serem removidas.

    Returns:
        pd.DataFrame: DataFrame modificado com as colunas removidas e a nova coluna.
    """

    existing_cols: List[str] = [col for col in cols_to_drop if col in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"Colunas removidas: {existing_cols}")
    else:
        print("Colunas não encontrada para remoção.")

    return df


def filter_column_availability(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    existing_cols = [col for col in columns if col in df.columns]

    if not existing_cols:
        print("Colunas não encontradas.")
        return df

    filtered_df = df.copy()

    def is_empty(val: Union[str, float, None]) -> bool:
        if pd.isna(val):
            return True
        if isinstance(val, str):
            return val.strip() == ""
        return False

    for column in existing_cols:
        series = filtered_df[column]

        mask = series.apply(is_empty)

        removed = mask.sum()
        filtered_df = filtered_df[~mask]

        print(
            f"{removed} linhas removidas por '{column}'. "
            f"Restam {len(filtered_df)} linhas."
        )

    return filtered_df



def gen_report(
    df: pd.DataFrame,
    output_file: str = "report.txt",
    save: bool = True,
    print_report: bool = False,
) -> None:
    """Gen info and summary to a text file if save is true"""
    str_analysis: str = (
        f"Dataset shape: {df.shape}\nColumns: {list(df.columns)}\n\nMissing Values:\n{df.isnull().sum()}\n\nNumeric Summary:\n{df.describe()}\n\nFirst 5 rows:\n{df.head()}\n"
    )
    if print_report:
        print(str_analysis)
    if not save:
        return
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str_analysis)
    print(f"\nReport salvo em {output_file}")


def main() -> None:


    # analise simples pra carregar e ver como estao os dados
    data_df = load_and_save_data()
    gen_report(data_df, output_file="raw_df.txt", save=False)

    # removendo categorias sem sentido
    categorias: List[str] = data_df["category"].dropna().unique().tolist()
    remove_set = set(["2016", "2015","bbc"])
    mask: pd.Series = pd.Series(True, index=data_df.index)
    mask &= ~data_df["category"].isin(remove_set)


    # adcionando dias da semana ao
    df_with_weekday = add_weekday_column(data_df)
    gen_report(df_with_weekday, output_file="df_with_weekday.txt", save=False)
    

    # remover subcategoria e links, subcategorias estao mal preenchidas
    # o link entregaria as categorias pq tem literalmente a categoria nele então não faria sentido a tarefa
    # não vou usar date pra analise, talvez weekday e já extrai essa informação 
    clean_df = clean(df_with_weekday, ["subcategory", "link","date"])
    gen_report(clean_df, output_file="clean_df.txt", save=False)

    #filtra as linhas que o texto ou titulo estao faltando
    filetered_df = filter_column_availability(clean_df, ["title", "text","category"])
    gen_report(filetered_df, output_file="filetered_df.txt", save=True)
    
  
    #salva os dados pra um csv
    filetered_df.to_csv("treated_data.csv", index=False)
    

if __name__ == "__main__":
    main()
