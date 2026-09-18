import joblib
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
    print(f"⚠️ Aviso: Modelo preditivo não encontrado ({e}). Usando fallback epidemiológico.")

def obter_rotulos_temporais():
    """Calcula automaticamente os nomes dos meses a partir da data atual do servidor"""
    agora = datetime.now()
    return {
        "atual": agora.strftime("%b/%Y"),
        "m1": (agora + relativedelta(months=1)).strftime("%b/%Y"),
        "m2": (agora + relativedelta(months=2)).strftime("%b/%Y"),
        "m3": (agora + relativedelta(months=3)).strftime("%b/%Y")
    }

def carregar_e_processar_dados():
    # Estrutura de dados epidemiológicos com histórico recente (lags) para alimentar o modelo
    dados_sp = [
        {"municipio": "São Paulo (Centro)", "lat": -23.5505, "lon": -46.6333, "casos_notificados": 1450, "populacao": 12300000, "incidencia_lag_1": 10.5, "incidencia_lag_2": 9.2},
        {"municipio": "Campinas", "lat": -22.9099, "lon": -47.0626, "casos_notificados": 3200, "populacao": 1210000, "incidencia_lag_1": 240.0, "incidencia_lag_2": 210.0},
        {"municipio": "Ribeirão Preto", "lat": -21.1704, "lon": -47.8103, "casos_notificados": 4100, "populacao": 710000, "incidencia_lag_1": 520.0, "incidencia_lag_2": 480.0},
        {"municipio": "São José dos Campos", "lat": -23.1896, "lon": -45.8841, "casos_notificados": 890, "populacao": 730000, "incidencia_lag_1": 110.0, "incidencia_lag_2": 95.0},
        {"municipio": "Sorocaba", "lat": -23.5015, "lon": -47.4526, "casos_notificados": 2100, "populacao": 695000, "incidencia_lag_1": 280.0, "incidencia_lag_2": 250.0},
        {"municipio": "Santos", "lat": -23.9608, "lon": -46.3339, "casos_notificados": 1150, "populacao": 433000, "incidencia_lag_1": 250.0, "incidencia_lag_2": 230.0},
        {"municipio": "São José do Rio Preto", "lat": -20.8113, "lon": -49.3758, "casos_notificados": 3800, "populacao": 469000, "incidencia_lag_1": 750.0, "incidencia_lag_2": 690.0},
        {"municipio": "Bauru", "lat": -22.3145, "lon": -49.0587, "casos_notificados": 1950, "populacao": 379000, "incidencia_lag_1": 480.0, "incidencia_lag_2": 430.0},
        {"municipio": "Piracicaba", "lat": -22.7253, "lon": -47.6492, "casos_notificados": 1600, "populacao": 407000, "incidencia_lag_1": 360.0, "incidencia_lag_2": 320.0},
        {"municipio": "Presidente Prudente", "lat": -22.1256, "lon": -51.3889, "casos_notificados": 2700, "populacao": 230000, "incidencia_lag_1": 1050.0, "incidencia_lag_2": 920.0},
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
    
    agora = datetime.now()
    mes_atual = agora.month
    rotulos_tempo = obter_rotulos_temporais()
    
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
        lag1 = item['incidencia_lag_1']
        lag2 = item['incidencia_lag_2']
        
        previsao = None

        # Tenta usar o modelo Random Forest carregado
        if modelo_preditivo:
            try:
                X_input = np.array([[taxa_atual, lag1, lag2, mes_atual]])
                pred_modelo = float(modelo_preditivo.predict(X_input)[0])
                
                # Validação de coerência: se a previsão do modelo não for desproporcionalmente achatada
                if pred_modelo > (taxa_atual * 0.15):
                    previsao = pred_modelo
            except Exception as err:
                print(f"Erro na predição do modelo para {item['municipio']}: {err}")

        # Fallback Epidemiológico Dinâmico (usado se o modelo .pkl não estiver carregado ou der valor estático)
        if previsao is None:
            # Tendência baseada na velocidade de variação recente (Lag 1 -> Taxa Atual)
            tendencia_recente = taxa_atual - lag1
            
            # Fator de sazonalidade por mês (meses de verão/outono têm peso maior)
            fator_sazonal = 1.08 if mes_atual in [1, 2, 3, 4, 5, 12] else 0.92
            
            # Cálculo da projeção dinâmica proporcional à taxa atual da cidade
            previsao = (taxa_atual + (tendencia_recente * 0.6)) * fator_sazonal

        # Lógica de Tendência Preditiva (Comparação entre Projeção e Atual)
        diferenca = previsao - taxa_atual
        
        if diferenca > 30:
            tendencia = "Alta Severa 📈"
        elif diferenca > 5:
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
    
    # Retorna o JSON unificado com K-Means + Previsão e Rótulos Automáticos de Tempo
    return jsonify({
        'rotulos_tempo': rotulos_tempo,
        'dados_sus': dados_municipios,
        'dados_treinamento': dados_municipios,
        'municipios': dados_municipios,
        'metricas_sus': {
            'total_casos': total_casos,
            'media_incidencia': media_incidencia,
            'municipio_critico': municipio_critico
        },
        'metricas': {
            'total_casos': total_casos,
            'media_incidencia': media_incidencia,
            'municipio_critico': municipio_critico
        }
    })

if __name__ == '__main__':
    app.run(debug=True)