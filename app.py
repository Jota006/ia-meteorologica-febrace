import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import json
import os

# Configuração da Página
st.set_page_config(
    page_title="IA Meteorológica - FEBRACE",
    page_icon="⛅",
    layout="wide"
)

st.title("⛅ Sistema de Inteligência Artificial Meteorológica")
st.markdown("**Projeto FEBRACE** — Monitoramento preditivo e adaptativo baseado em Regressão Linear e Coleta Automática em Nuvem.")

# Carrega os dados coletados pelo robô na nuvem (ou usa padrão se não existir ainda)
arquivo_json = "dados_clima.json"

if os.path.exists(arquivo_json):
    with open(arquivo_json, "r") as f:
        dados = json.load(f)
        df = pd.DataFrame(dados)
else:
    # Dados iniciais de segurança caso o robô ainda não tenha rodado
    df = pd.DataFrame({
        'hora': [8, 10, 12, 14, 16],
        'temperatura': [22.5, 24.0, 27.5, 29.1, 28.0],
        'umidade': [75, 70, 55, 48, 52]
    })

# Verifica se há dados suficientes
if len(df) > 0:
    ultima_hora = int(df['hora'].iloc[-1])
    ultima_temp = df['temperatura'].iloc[-1]
    ultima_umid = df['umidade'].iloc[-1]

    # Indicadores Atuais
    st.subheader("📍 Status Atual do Monitoramento (Escola)")
    col1, col2, col3 = st.columns(3)

    col1.metric(label="🕒 Última Leitura", value=f"{ultima_hora}:00 hrs")
    col2.metric(label="🌡️ Temperatura Atual", value=f"{ultima_temp} °C")
    col3.metric(label="💧 Umidade Relativa", value=f"{ultima_umid} %")

    st.divider()

    # Treinamento da Inteligência Artificial em tempo real
    X = df[['hora']]
    modelo_temp = LinearRegression().fit(X, df['temperatura'])
    modelo_umid = LinearRegression().fit(X, df['umidade'])

    # Seção de Previsões da IA
    st.subheader("🤖 Estimativas da Inteligência Artificial para as Próximas Horas")
    col_prev1, col_prev2, col_prev3 = st.columns(3)
    colunas_pred = [col_prev1, col_prev2, col_prev3]

    for i, coluna in enumerate(colunas_pred, start=1):
        proxima_hora = (ultima_hora + i) % 24
        p_temp = modelo_temp.predict([[proxima_hora]])[0]
        p_umid = modelo_umid.predict([[proxima_hora]])[0]
        
        with coluna:
            st.info(f"**Às {proxima_hora}:00 hrs**")
            st.write(f"Temp Prevista: **{p_temp:.1f}°C**")
            st.write(f"Umidade Prevista: **{p_umid:.1f}%**")

    st.divider()

    # Gráfico Interativo
    st.subheader("📈 Evolução Térmica e Tendência do Modelo")
    fig = px.line(df, x='hora', y='temperatura', markers=True, title="Histórico de Temperaturas Coletadas")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Aguardando os primeiros dados do sistema...")
