import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
   
    # 6
    #df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size()

    # 7
    g = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count')
    g.set_ylabels('total')


    # 8
    fig = g.fig
    

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11. Limpiar los datos en el df_heat filtrando segmentos de pacientes incorrectos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]  # Hacer una copia explícita del DataFrame para evitar SettingWithCopyWarning

    # 12. Calcular la matriz de correlación usando los datos limpios
    corr = df_heat.corr().round(1)

    # 13. Generar una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=int))

    # 14. Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Graficar la matriz de correlación utilizando sns.heatmap()
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm', mask=mask, square=True, linewidths=0.5, ax=ax)

    # 16. No modificar las siguientes dos líneas
    fig.savefig('heatmap.png')
    return fig
