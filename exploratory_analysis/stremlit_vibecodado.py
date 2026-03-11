# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List
import os


@st.cache_data
def load_data(filepath: str = "treated_data.csv") -> Optional[pd.DataFrame]:
    """Carrega os dados tratados a partir de um arquivo CSV."""
    if not os.path.exists(filepath):
        st.error(f"Arquivo {filepath} não encontrado. Por favor, salve seu DataFrame tratado como CSV.")
        return None
    df = pd.read_csv(filepath)
    return df


def main() -> None:
    st.set_page_config(page_title="Dados Tratados - Folha/UOL", layout="wide")
    st.title("📰 Dados Tratados - Notícias Folha/UOL")
    st.markdown("---")

    # Carregar dados
    df = load_data()
    if df is None:
        st.stop()

    # Sidebar com informações
    st.sidebar.header("Informações")
    st.sidebar.write(f"**Linhas:** {df.shape[0]:,}")
    st.sidebar.write(f"**Colunas:** {df.shape[1]}")
    st.sidebar.write(f"**Colunas disponíveis:** {list(df.columns)}")

    # Verificar colunas essenciais
    colunas_esperadas = ["title", "text", "category", "weekday"]
    for col in colunas_esperadas:
        if col not in df.columns:
            st.sidebar.warning(f"Coluna '{col}' não encontrada. Alguns gráficos podem não funcionar.")

    st.markdown("## 📊 Estatísticas Gerais")

    # Métricas principais em colunas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de artigos", f"{df.shape[0]:,}")
    with col2:
        if "category" in df.columns:
            st.metric("Categorias distintas", df["category"].nunique())
    with col3:
        if "weekday" in df.columns:
            st.metric("Dias da semana", df["weekday"].nunique())
    with col4:
        if "text" in df.columns:
            # Calcular tamanho médio do texto
            text_lengths = df["text"].astype(str).str.len()
            st.metric("Tamanho médio do texto", f"{text_lengths.mean():.0f} caracteres")

    st.markdown("---")

    # Gráficos
    col_esq, col_dir = st.columns(2)

    with col_esq:
        if "category" in df.columns:
            st.subheader("Distribuição por Categoria")
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["categoria", "quantidade"]
            fig_cat = px.bar(
                cat_counts.head(20),  # Top 20 para não poluir
                x="categoria",
                y="quantidade",
                title="Top 20 categorias mais frequentes",
                color="quantidade",
                color_continuous_scale="viridis"
            )
            fig_cat.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Coluna 'category' não disponível.")

    with col_dir:
        if "weekday" in df.columns:
            st.subheader("Distribuição por Dia da Semana")

            ordem_dias = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]

        # Se forem números
        if df["weekday"].dtype in ["int64", "float64"]:
            dias_map = {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            }
            weekday_series = df["weekday"].map(dias_map)
        else:
            weekday_series = df["weekday"].str.strip().str.title()

        weekday_counts = (
            weekday_series
            .value_counts()
            .reindex(ordem_dias, fill_value=0)
            .reset_index()
        )

        weekday_counts.columns = ["dia", "quantidade"]

        fig_dia = px.bar(
            weekday_counts,
            x="dia",
            y="quantidade",
            title="Artigos por dia da semana",
            color="quantidade",
            color_continuous_scale="plasma"
        )

        st.plotly_chart(fig_dia, use_container_width=True)

    st.markdown("---")

    # Análise de texto
    if "text" in df.columns:
        st.subheader("📏 Distribuição do tamanho do texto")
        text_lengths = df["text"].astype(str).str.len()
        fig_hist = px.histogram(
            text_lengths,
            nbins=50,
            title="Histograma do número de caracteres por artigo",
            labels={"value": "Número de caracteres", "count": "Frequência"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Estatísticas descritivas do tamanho do texto
        st.write("**Estatísticas do tamanho do texto:**")
        stats = text_lengths.describe().to_frame().T
        st.dataframe(stats)

    st.markdown("---")

    # Amostra dos dados
    with st.expander("🔍 Visualizar amostra dos dados (primeiras 100 linhas)"):
        st.dataframe(df.head(100))

    # Download dos dados tratados
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar dados tratados como CSV",
        data=csv,
        file_name="dados_tratados.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()