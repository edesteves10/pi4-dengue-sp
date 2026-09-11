import joblib
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Tenta carregar o modelo preditivo previamente treinado
try:
    modelo_preditivo = joblib.load('modelo_dengue_preditivo.pkl')
    print("✅ Modelo preditivo Random Forest carregado com sucesso!")
except Exception as e:
    modelo_preditivo = None
    print(f"⚠️ Aviso: Modelo preditivo não encontrado ({e}). Usando fallback.")

def carregar_e_processar_dados():
    # Estrutura de dados epidemiológicos
    dados_sp = [
        {"municipio": "São Paulo (Centro)", "lat": -23.5505, "lon": -46.6333, "casos_notificados": 1450, "populacao": 12300000},
        {"municipio": "Campinas", "lat": -22.9099, "lon": -47.0626, "casos_notificados": 3200, "populacao": 1210000},
        {"municipio": "Ribeirão Preto", "lat": -21.1704, "lon": -47.8103, "casos_notificados": 4100, "populacao": 710000},
        {"municipio": "São José dos Campos", "lat": -23.1896, "lon": -45.8841, "casos_notificados": 890, "populacao": 730000},
        {"municipio": "Sorocaba", "lat": -23.5015, "lon": -47.4526, "casos_notificados": 2100, "populacao": 695000},
        {"municipio": "Santos", "lat": -23.9608, "lon": -46.3339, "casos_notificados": 1150, "populacao": 433000},
        {"municipio": "São José do Rio Preto", "lat": -20.8113, "lon": -49.3758, "casos_notificados": 3800, "populacao": 469000},
        {"municipio": "Bauru", "lat": -22.3145, "lon": -49.0587, "casos_notificados": 1950, "populacao": 379000},
        {"municipio": "Piracicaba", "lat": -22.7253, "lon": -47.6492, "casos_notificados": 1600, "populacao": 407000},
        {"municipio": "Presidente Prudente", "lat": -22.1256, "lon": -51.3889, "casos_notificados": 2700, "populacao": 230000},
    ]

    df = pd.DataFrame(dados_sp)
    
    # Métrica de incidência por 100 mil habitantes
    df['incidencia_100k'] = (df['casos_notificados'] / df['populacao']) * 100000
    df['incidencia_100k'] = df['incidencia_100k'].round(2)
    
    # Machine Learning - K-Means (Classificação Atual)
    X = df[['incidencia_100k']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    centros = kmeans.cluster_centers_.flatten()
    ordem_clusters = sorted(range(len(centros)), key=lambda k: centros[k])
    mapeamento_risco = {ordem_clusters[0]: 'Baixo', ordem_clusters[1]: 'Médio', ordem_clusters[2]: 'Alto'}
    df['nivel_risco'] = df['cluster'].map(mapeamento_risco)
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dados')
def api_dados():
    df = carregar_e_processar_dados()
    mes_atual = datetime.now().month
    
    # Cálculo das Métricas Gerais
    total_casos = int(df['casos_notificados'].sum())
    media_incidencia = round(float(df['incidencia_100k'].mean()), 2)
    
    municipio_critico_row = df.loc[df['incidencia_100k'].idxmax()]
    municipio_critico = {
        'nome': municipio_critico_row['municipio'],
        'incidencia': float(municipio_critico_row['incidencia_100k'])
    }
    
    # Processamento Preditivo para cada município
    dados_municipios = []
    lista_registros = df.to_dict(orient='records')

    for item in lista_registros:
        taxa_atual = item['incidencia_100k']
        
        if modelo_preditivo:
            # Puxa os dados históricos (ou aproximação baseada na taxa atual)
            lag1 = item.get('incidencia_lag_1', taxa_atual)
            lag2 = item.get('incidencia_lag_2', taxa_atual)
            
            X_input = np.array([[taxa_atual, lag1, lag2, mes_atual]])
            previsao = float(modelo_preditivo.predict(X_input)[0])
        else:
            previsao = taxa_atual * 1.05  # Fallback simulado se o .pkl não estiver carregado

        # Lógica de Tendência Preditiva
        diferenca = previsao - taxa_atual
        if diferenca > 40:
            tendencia = "Alta Severa 📈"
        elif diferenca > 10:
            tendencia = "Aumento Moderado ↗️"
        else:
            tendencia = "Estável / Queda ↘️"

        dados_municipios.append({
            'municipio': item['municipio'],
            'lat': item['lat'],
            'lon': item['lon'],
            'casos_notificados': item['casos_notificados'],
            'incidencia_100k': taxa_atual,
            'nivel_risco': item['nivel_risco'],
            'incidencia_prevista_100k': round(max(0, previsao), 2),
            'tendencia': tendencia
        })
    
    # Retorna o JSON unificado com K-Means + Previsão Random Forest
    return jsonify({
        'municipios': dados_municipios,
        'metricas': {
            'total_casos': total_casos,
            'media_incidencia': media_incidencia,
            'municipio_critico': municipio_critico
        }
    })

if __name__ == '__main__':
    app.run(debug=True)