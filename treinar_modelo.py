import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def treinar_e_salvar_modelo():
    print("⏳ Carregando base histórica de dados (2020-2026)...")
    # Substitua pelo caminho do seu arquivo CSV compilado do SUS/SINAN
    df = pd.read_csv('dados_dengue_2020_2026.csv')

    # Garantir ordenação temporal por município
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values(['municipio', 'data'])

    # Cálculo da taxa de incidência por 100 mil habitantes
    df['incidencia'] = (df['casos_notificados'] / df['populacao']) * 100000

    print("⚙️ Criando variáveis de atraso temporal (lags)...")
    # Engenharia de Features Temporais
    df['incidencia_atual'] = df['incidencia']
    df['incidencia_lag_1'] = df.groupby('municipio')['incidencia'].shift(1)  # Mês t-1
    df['incidencia_lag_2'] = df.groupby('municipio')['incidencia'].shift(2)  # Mês t-2
    df['mes_do_ano'] = df['data'].dt.month

    # Variável Alvo (Target): Incidência do próximo mês (t+1)
    df['alvo_proximo_mes'] = df.groupby('municipio')['incidencia'].shift(-1)

    # Remover linhas sem histórico suficiente
    df_modelo = df.dropna(subset=['incidencia_lag_1', 'incidencia_lag_2', 'alvo_proximo_mes'])

    # Variáveis de entrada (X) e alvo (y)
    X = df_modelo[['incidencia_atual', 'incidencia_lag_1', 'incidencia_lag_2', 'mes_do_ano']]
    y = df_modelo['alvo_proximo_mes']

    # Treinamento do Modelo Random Forest
    print("🤖 Treinando o modelo Random Forest Regressor...")
    modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X, y)

    # Avaliação simples de métricas
    previsoes_treino = modelo.predict(X)
    r2 = r2_score(y, previsoes_treino)
    print(f"📊 Acurácia do modelo (R² Score no treino): {r2:.2f}")

    # Exportar o arquivo binário do modelo
    joblib.dump(modelo, 'modelo_dengue_preditivo.pkl')
    print("✅ Arquivo 'modelo_dengue_preditivo.pkl' gerado com sucesso!")

if __name__ == '__main__':
    treinar_e_salvar_modelo()