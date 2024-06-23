import pandas as pd
from pickle import load
obesity_clusters_kmeans = load(open("dados/obesity_cluster.pkl", "rb"))
normalizador = load(open("dados/normalizador.pkl", "rb"))

colunas_categoricas = load(open("dados/colunas_Categoricas.pkl", "rb"))
teste_instancia = ['Female', 21, 1.62, 64, 'no', 'no', 2, 3, 'Sometimes', 'no', 2, 'no', 0, 1, 'no', 'Public_Transportation', 'Normal_Weight']
pd.set_option('display.max_columns', None)

df_teste = pd.DataFrame([teste_instancia], columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'])

dados_categoricos_normalizados = pd.get_dummies(data=df_teste[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']], dtype=int)
dados_numericos = df_teste.drop(columns=['Gender','family_history_with_overweight', 'FAVC', 'CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad' ])
dados_numericos_normalizados = normalizador.transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE'])
dados_categoricos = pd.DataFrame(columns=colunas_categoricas)

dados_completos = pd.concat([dados_categoricos, dados_categoricos_normalizados], axis=0)

dados_completos = dados_completos.where(pd.notna(dados_completos), other=0)
dados_completos = dados_numericos_normalizados.join(dados_completos, how='left')


dados_normalizados_final_legiveis = normalizador.inverse_transform(dados_numericos_normalizados)
dados_categoricos_legiveis = dados_categoricos_normalizados
dados_normalizados_final_legiveis = pd.DataFrame(data= dados_normalizados_final_legiveis, columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']).join(dados_categoricos_normalizados)

print(f"índice do grupo da nova instancia{obesity_clusters_kmeans.predict(dados_completos.values)}")
print(f"Centroide da nova instancia: {obesity_clusters_kmeans.cluster_centers_[obesity_clusters_kmeans.predict(dados_completos)]}")


colunas = ['Gender_', 'family_history_with_overweight_', 'FAVC_', 'CAEC_', 'SMOKE_', 'SCC_', 'CALC_', 'MTRANS_', 'NObeyesdad_']

for coluna in dados_categoricos_normalizados.columns:
    for prefixo in colunas:
        if coluna.startswith(prefixo):
            valor = coluna[len(prefixo):].strip()
            dados_categoricos_normalizados.rename(columns={coluna: prefixo.rstrip('_')}, inplace=True)
            dados_categoricos_normalizados[prefixo.rstrip('_')] = valor
            break

df_final = pd.concat([dados_categoricos_normalizados, dados_normalizados_final_legiveis], axis=1)

df_final = df_final[df_teste.columns]

print(df_final)