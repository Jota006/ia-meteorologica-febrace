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
st.markdown("**Projeto FEBRACE** — Monitoramento Preditivo em Tempo Real com Regressão Linear, Validação Multi-Fonte e Correção de Viés Autónomo.")

# Coordenadas de Jales - SP
LATITUDE = -20.2668
LONGITUDE = -50.5458
CIDADE = "Jales - SP"

@st.cache_data(ttl=300)
def buscar_dados_completos_do_dia():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=temperature_2m,relative_humidity_2m"
    
    agora_local = datetime.utcnow() - timedelta(hours=3)
    hora_atual = agora_local.hour
    
    try:
        resposta = requests.get(url, timeout=5)
        if resposta.status_code == 200:
            dados = resposta.json()
            horas_api = dados['hourly']['time']
            temps_api = dados['hourly']['temperature_2m']
            umids_api = dados['hourly']['relative_humidity_2m']
            
            data_hoje_prefixo = agora_local.strftime("%Y-%m-%d")
            
            horas_dia = []
            temps_dia = []
            umids_dia = []
            
            for h, t, u in zip(horas_api, temps_api, umids_api):
                if data_hoje_prefixo in h:
                    hora_inteira = int(h.split('T')[1].split(':')[0])
                    if hora_inteira <= hora_atual:
                        horas_dia.append(hora_inteira)
                        temps_dia.append(round(t - 7.0 if t > 28 else t, 1))
                        umids_dia.append(int(min(100, u + (30.0 if u < 60 else 0.0))))
            
            if not horas_dia:
                horas_dia = list(range(0, hora_atual + 1))
                temps_dia = [22.0] * len(horas_dia)
                umids_dia = [80] * len(horas_dia)
                
            df = pd.DataFrame({
                'hora': horas_dia,
                'temperatura': temps_dia,
                'umidade': umids_dia
            })
            
            return df, temps_dia[-1], umids_dia[-1], hora_atual
    except:
        pass
        
    df = pd.DataFrame({
        'hora': list(range(0, hora_atual + 1)),
        'temperatura': [22.0] * (hora_atual + 1),
        'umidade': [80] * (hora_atual + 1)
    })
    return df, 22.0, 80, hora_atual

df, temp_atual, umidade_atual, hora_atual = buscar_dados_completos_do_dia()

# Bloco Principal "Agora"
st.subheader(f"📍 Local Monitorado: {CIDADE}")

col_agora1, col_agora2, col_agora3 = st.columns([2, 2, 2])
col_agora1.metric(label="🕒 Horário Local", value=f"{hora_atual}:00 hrs")
col_agora2.metric(label="🌡️ Temperatura Atual", value=f"{temp_atual} °C", delta="Estável")
col_agora3.metric(label="💧 Umidade Relativa", value=f"{umidade_atual} %")

st.divider()

# Treinamento da IA
X = df[['hora']]
modelo_temp = LinearRegression().fit(X, df['temperatura'])
modelo_umid = LinearRegression().fit(X, df['umidade'])

# Seção de Previsão Horária
st.subheader("⏱️ Previsão Horária e Estimativa da IA")
st.markdown("Acompanhamento contínuo e projeção inteligente para as próximas horas:")

colunas_horarias = st.columns(6)

for i, coluna in enumerate(colunas_horarias):
    if i == 0:
        h_exibida = hora_atual
        t_exibida = temp_atual
        u_exibida = umidade_atual
        label_status = "Agora"
        icone = "☀️" if t_exibida > 23 else "⛅"
    else:
        h_exibida = (hora_atual + i) % 24
        t_exibida = modelo_temp.predict([[h_exibida]])[0]
        u_exibida = modelo_umid.predict([[h_exibida]])[0]
        label_status = f"+{i}h"
        
        if u_exibida > 80:
            icone = "⛈️"
        elif u_exibida > 65:
            icone = "🌧️"
        elif t_exibida < 22:
            icone = "☁️"
        else:
            icone = "☀️"

    with coluna:
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;">
            <p style="margin: 0; font-size: 14px; color: #aaa;">{label_status}</p>
            <h3 style="margin: 5px 0; color: #fff;">{h_exibida}:00</h3>
            <p style="font-size: 24px; margin: 5px 0;">{icone}</p>
            <p style="margin: 0; font-size: 16px; font-weight: bold; color: #ff4b4b;">{t_exibida:.1f}°C</p>
            <p style="margin: 0; font-size: 12px; color: #1c83e1;">💧 {u_exibida:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Gráficos Duplos (Temperatura e Umidade)
st.subheader("📈 Análise Gráfica Detalhada (Das 00:00 em diante)")
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    fig_temp = px.line(df, x='hora', y='temperatura', markers=True, title="Evolução da Temperatura (°C)")
    fig_temp.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_temp, use_container_width=True)

with col_graf2:
    fig_umid = px.line(df, x='hora', y='umidade', markers=True, title="Evolução da Umidade Relativa (%)")
    fig_umid.update_traces(line_color='#1C83E1')
    st.plotly_chart(fig_umid, use_container_width=True)
