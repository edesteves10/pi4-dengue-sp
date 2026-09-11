from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

def carregar_e_processar_dados():
    # Estrutura consolidada com dados de amostragem epidemiológica
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
    
    # Métrica oficial de saúde: taxa de incidência por 100k habitantes
    df['incidencia_100k'] = (df['casos_notificados'] / df['populacao']) * 100000
    df['incidencia_100k'] = df['incidencia_100k'].round(2)
    
    # Aplicação do Machine Learning (K-Means com 3 clusters)
    X = df[['incidencia_100k']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Ordenação dos clusters para mapear níveis de risco (0: Baixo, 1: Médio, 2: Alto)
    centros = kmeans.cluster_centers_.flatten()
    ordem_clusters = numpy_args = sorted(range(len(centros)), key=lambda k: centros[k])
    mapeamento_risco = {ordem_clusters[0]: 'Baixo', ordem_clusters[1]: 'Médio', ordem_clusters[2]: 'Alto'}
    
    df['nivel_risco'] = df['cluster'].map(mapeamento_risco)
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dados')
def api_dados():
    df = carregar_e_processar_dados()
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)