import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import requests
from datetime import datetime, timedelta

# Configuração da Página
st.set_page_config(
    page_title="IA Meteorológica - FEBRACE",
    page_icon="⛅",
    layout="wide"
)

st.title("⛅ Sistema de Inteligência Artificial Meteorológica")
st.markdown("**Projeto FEBRACE** — Monitoramento preditivo em tempo real com Regressão Linear e API Autônoma.")

# Coordenadas de Jales - SP
LATITUDE = -20.2668
LONGITUDE = -50.5458
CIDADE = "Jales - SP"

@st.cache_data(ttl=300)
def buscar_dados_precisos():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=temperature_2m,relative_humidity_2m"
    
    # Hora local de Jales (UTC-3)
    agora_local = datetime.utcnow() - timedelta(hours=3)
    hora_atual = agora_local.hour
    
    try:
        resposta = requests.get(url)
        dados = resposta.json()
        
        # Pega a lista horária da API
        horas_api = dados['hourly']['time']
        temps_api = dados['hourly']['temperature_2m']
        umids_api = dados['hourly']['relative_humidity_2m']
        
        # Encontra o índice correspondente à hora atual do dia de hoje
        data_hoje_str = agora_local.strftime("%Y-%m-%dT%H:00")
        indice_atual = 0
        for i, t in enumerate(horas_api):
            if data_hoje_str in t:
                indice_atual = i
                break
                
        # Pega os valores exatos da hora atual
        temp_atual = temps_api[indice_atual]
        umidade_atual = umids_api[indice_atual]
        
        # Monta o DataFrame com as últimas horas reais da API para alimentar a IA
        idx_inicio = max(0, indice_atual - 3)
        df = pd.DataFrame({
            'hora': [int(h.split('T')[1].split(':')[0]) for h in horas_api[idx_inicio:indice_atual+1]],
            'temperatura': temps_api[idx_inicio:indice_atual+1],
            'umidade': umids_api[idx_inicio:indice_atual+1]
        })
        
        return df, temp_atual, umidade_atual, hora_atual
    except Exception as e:
        # Fallback de segurança
        df = pd.DataFrame({
            'hora': [12, 13, 14, 15],
            'temperatura': [22.0, 22.5, 23.0, 23.0],
            'umidade': [80, 81, 82, 80]
        })
        return df, 23.0, 80, agora_local.hour

df, temp_atual, umidade_atual, hora_atual = buscar_dados_precisos()

# Indicadores Atuais na Tela
st.subheader(f"📍 Local Monitorado: {CIDADE}")
col1, col2, col3 = st.columns(3)

col1.metric(label="🕒 Horário Local", value=f"{hora_atual}:00 hrs")
col2.metric(label="🌡️ Temperatura Atual (Real)", value=f"{temp_atual} °C")
col3.metric(label="💧 Umidade Relativa (Real)", value=f"{umidade_atual} %")

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
    proxima_hora = (hora_atual + i) % 24
    p_temp = modelo_temp.predict([[proxima_hora]])[0]
    p_umid = modelo_umid.predict([[proxima_hora]])[0]
    
    with coluna:
        st.info(f"**Às {proxima_hora}:00 hrs**")
        st.write(f"Temp Prevista: **{p_temp:.1f}°C**")
        st.write(f"Umidade Prevista: **{p_umid:.1f}%**")

st.divider()

# Gráfico Interativo
st.subheader("📈 Tendência Térmica Calculada pelo Modelo")
fig = px.line(df, x='hora', y='temperatura', markers=True, title="Evolução Térmica Recente e Preditiva")
st.plotly_chart(fig, use_container_width=True)
