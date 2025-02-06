import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import kstest, shapiro, anderson

import seaborn as sns
from scipy.stats import norm
from scipy.stats import pearsonr
from itertools import combinations
import plotly.express as px
from PIL import Image
import os

# Crear un DataFrame de pandas
df = pd.read_csv('Cerebrovasculares.csv')


#Crean las tablas con los datos de los estadíticos de centro y de dispersión
variables = ['Edad', 'Avg_Glucosa', 'IMC']
tablaEstCentro = pd.DataFrame(columns= ['Media', 'Mediana', 'Primer_Cuartil', 'Tercer_Cuartil'], index= variables)
tablaEstDispersion = pd.DataFrame(columns= ['Valor_max', 'Valor_min', 'Rango', 'Varianza', 'Desv_Std', 'CV%', 'Rango_Intercuartilico'], index= variables)

for elemento in variables:
    tablaEstCentro.loc[elemento, 'Media'] = round(df[elemento].mean(), 2)
    tablaEstCentro.loc[elemento, 'Mediana'] = round(df[elemento].median(),2)
    tablaEstCentro.loc[elemento, 'Primer_Cuartil'] = round(df[elemento].quantile(0.25), 2)
    tablaEstCentro.loc[elemento, 'Tercer_Cuartil'] = round(df[elemento].quantile(0.75),2)
    
    tablaEstDispersion.loc[elemento, 'Valor_max'] = df[elemento].max()
    tablaEstDispersion.loc[elemento, 'Valor_min'] = df[elemento].min()
    tablaEstDispersion.loc[elemento, 'Rango'] = round(tablaEstDispersion.loc[elemento, 'Valor_max'] - tablaEstDispersion.loc[elemento, 'Valor_min'],2)
    tablaEstDispersion.loc[elemento, 'Varianza'] = round(df[elemento].var(), 2)
    tablaEstDispersion.loc[elemento, 'Desv_Std'] = round(df[elemento].std(ddof=1), 2)
    tablaEstDispersion.loc[elemento, 'CV%'] = round((tablaEstDispersion.loc[elemento, 'Desv_Std'] / tablaEstCentro.loc[elemento, 'Media']) * 100, 2)
    tablaEstDispersion.loc[elemento, 'Rango_Intercuartilico'] = round((tablaEstCentro.at[elemento, 'Tercer_Cuartil'] - tablaEstCentro.at[elemento, 'Primer_Cuartil']), 2)

# Guarda la tabla de los estadísticos de centro como imágenes
fig, ax = plt.subplots(figsize=(10, 2)) 
ax.axis('tight')
ax.axis('off')
tablaEstCentro_table = ax.table(cellText=tablaEstCentro.values, colLabels=tablaEstCentro.columns, rowLabels=tablaEstCentro.index, cellLoc='center', loc='center')
tablaEstCentro_table.auto_set_font_size(False)
tablaEstCentro_table.set_fontsize(10)
tablaEstCentro_table.scale(1.2, 2) 

# Se le da un color gris de fondo a la cabecera de las filas y las columnas
for key, cell in tablaEstCentro_table.get_celld().items():
    if key[0] == 0 or key[1] == -1:
        cell.set_facecolor('lightgray')

plt.savefig('./img/Tablas/Estadísticos_de_centro.png', dpi=300, bbox_inches='tight')


# Guarda la tabla de los estadísticos de dispersión como imágenes
fig, ax = plt.subplots(figsize=(12, 2)) 
ax.axis('tight')
ax.axis('off')
tablaEstDispersion_table = ax.table(cellText=tablaEstDispersion.values, colLabels=tablaEstDispersion.columns, rowLabels=tablaEstDispersion.index, cellLoc='center', loc='center')
tablaEstDispersion_table.auto_set_font_size(False)
tablaEstDispersion_table.set_fontsize(10)
tablaEstDispersion_table.scale(1.2, 2)  # Increase the width and height of the table

# Se le da un color gris de fondo a la cabecera de las filas y las columnas
for key, cell in tablaEstDispersion_table.get_celld().items():
    if key[0] == 0 or key[1] == -1:
        cell.set_facecolor('lightgray')

plt.savefig('./img/Tablas/Estadísticos_de_dispersión.png', dpi=300, bbox_inches='tight')


#Crear gráficos pastel de las variables cualitativas

#Se crea un método que hace el gráfico pastel de las variables bernulli del dataset
def crear_grafico_pastel(variable, label1, label2):
    df_map = df[variable].map({0: label1, 1: label2})
    tabla = pd.crosstab(index = df_map, columns='Frecuencia')
    
    plt.figure(figsize=(6,6))
    colors = sns.color_palette('pastel')[0:2]
    plt.pie(tabla['Frecuencia'], labels=[label1, label2], autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    plt.title("Gráfico de Pastel de la variable " + variable)

    plt.savefig("./img/GráficosPastel/grafico_pastel_" + variable + ".png", dpi = 300)

# El primer parametro es el nombre de la variable del dataset, segundo la variable 
# categórica que se codifica con 0 y tercero la que se codifica con 1.
crear_grafico_pastel('Sex', 'Femenino', 'Masculino')
crear_grafico_pastel('Hipertension', 'No hipertenso', 'Hipertenso')
crear_grafico_pastel('Casado', 'Soltero', 'casado')
crear_grafico_pastel('Cardiopatia', 'No presenta cardiopatía', 'Presenta cardiopatía')
crear_grafico_pastel('Tipo_Residencia', 'Urbana', 'Rural')
crear_grafico_pastel('Fumar', 'No fuma', 'fuma')
crear_grafico_pastel('Accidentes', 'Ha tenido accidente cerebrovascular', 'No ha tenido')

# df_map = df['Tipo_Trabajo'].map({
#     0: 'Nunca ha trabajado', 
#     1: 'Trabajo eventual', 
#     2: 'Trabajo estatal', 
#     3: 'Trabajo por cuenta propia'
# })
# tabla = pd.crosstab(index=df_map, columns='Frecuencia')

# plt.figure(figsize=(6, 6))
# # colors = sns.color_palette('pastel')[0:4]
# plt.pie(
#     tabla['Frecuencia'], 
#     labels=[
#         'Nunca ha trabajado', 
#         'Trabajo eventual', 
#         'Trabajo estatal', 
#         'Trabajo por cuenta propia'
#     ],
#     autopct='%1.1f%%', 
#     startangle=90, 
#     # colors=colors, 
#     wedgeprops={'edgecolor': 'black'}
# )
# plt.title("Gráfico de Pastel de la variable Tipo de trabajo")

# plt.savefig("./img/grafico_pastel_Tipo_Trabajo.png", dpi = 300)
#Crear histogramas de las variables cuantitativas

def Crear_Histograma (variable, color_barras):
    plt.figure(figsize=(10,6))
    conteo, bins,_ = plt.hist(df[variable], bins=10, density=True, alpha=0.6, color= color_barras, edgecolor='black', label='Histograma de la variable ' + variable)
    media, dev_std = np.mean(df[variable]), np.std(df[variable])
    x = np.linspace(min(df[variable]), max(df[variable]), 1000)
    densidad = norm.pdf(x, media, dev_std)
    plt.plot(x, densidad, color='red', lw=2, label=f'Distribución de los datos (media = {media:.2f}, std = {dev_std:.2f})')

    plt.title('Histograma de la variable ' + variable, fontsize=14)
    plt.xlabel(variable, fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend()
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.75)

    plt.savefig("./img/Histogramas/Histograma_" + variable + ".png", dpi = 300)


Crear_Histograma('Edad', 'lightgreen' )
Crear_Histograma('IMC', 'lightgray' )
Crear_Histograma('Avg_Glucosa', 'gold' )



def Crear_Boxplot (variable, color):
    # Crear el boxplot IMC
    plt.figure(figsize=(10, 6))
    box_width = 0.3  # Ancho específico de la caja
    plt.boxplot(df[variable], vert=False, patch_artist=True, widths=box_width, boxprops=dict(facecolor="white"), zorder=1)
    plt.title(f'Boxplot de la variable ' + variable)
    plt.xlabel(variable)
    plt.grid(False)

    # Añadir stripchart usando plt
    y_values = np.random.uniform(1 - box_width / 2, 1 + box_width / 2, size=len(df[variable]))
    plt.scatter(df[variable], y_values, alpha=0.6, color=color, zorder=2)

    plt.savefig("./img/Boxplot/Boxplt_" + variable + ".png", dpi = 300)

Crear_Boxplot('Edad', 'blue')
Crear_Boxplot('IMC', 'red')
Crear_Boxplot('Avg_Glucosa', 'yellow')



def test_normalidad_Kolmogorov_Smirnov(variable, nivel_significancia=0.05):
    # Ajustar la media y desviación estándar de los datos a una distribución normal
    mean, std = df[variable].mean(), df[variable].std()

    # Prueba de K-S para la distribución normal
    stat, p_value = kstest(df[variable], 'norm', args=(mean, std))
    conclusion = 'Normal' if p_value > nivel_significancia else 'No Normal'
    
    return [stat, p_value, conclusion]

def test_normalidad_Shapiro_Wilk(variable, nivel_significancia=0.05):
    stat, p_value = shapiro(df[variable])
    conclusion = 'Normal' if p_value > nivel_significancia else 'No Normal'
    
    return [stat, p_value, conclusion]

def test_Anderson_Darling(variable):
    result = anderson(df[variable])
    stat = result.statistic
    p_value = '-'  # Anderson-Darling no proporciona un p-valor directamente
    conclusion = 'Normal' if all(stat < cv for cv in result.critical_values) else 'No Normal'
    
    return [stat, p_value, conclusion]

def crear_tabla_test_normalidad(variable):
    pruebas = ['Kolmogorov Smirnov', 'Shapiro_Wilk', 'Anderson Darling']
    columnas = ['Estadístico', 'p-valor', 'Conclusión']
    tabla = pd.DataFrame(columns= columnas, index= pruebas)
    # Prueba de Kolmogorov-Smirnov
    stat, p_value, conclusion = test_normalidad_Kolmogorov_Smirnov(variable)
    tabla.loc['Kolmogorov Smirnov'] = [stat, p_value, conclusion]

    # Prueba de Shapiro-Wilk
    stat, p_value, conclusion = test_normalidad_Shapiro_Wilk(variable)
    tabla.loc['Shapiro_Wilk'] = [stat, p_value, conclusion]

    # Prueba de Anderson-Darling
    stat, p_value, conclusion = test_Anderson_Darling(variable)
    tabla.loc['Anderson Darling'] = [stat, p_value, conclusion]

    # Guardar la tabla de pruebas de normalidad como imagen
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    tabla_test_normalidad = ax.table(cellText=tabla.values, colLabels=tabla.columns, rowLabels=tabla.index, cellLoc='center', loc='center')
    tabla_test_normalidad.auto_set_font_size(False)
    tabla_test_normalidad.set_fontsize(10)
    tabla_test_normalidad.scale(1.2, 2.5)

    # Se le da un color gris de fondo a la cabecera de las filas y las columnas
    for key, cell in tabla_test_normalidad.get_celld().items():
        if key[0] == 0 or key[1] == -1:
            cell.set_facecolor('lightgray')

    plt.savefig('./img/Tablas/test_normalidad_' + variable + '.png', dpi=300, bbox_inches='tight')


crear_tabla_test_normalidad('Edad')
crear_tabla_test_normalidad('Avg_Glucosa')
crear_tabla_test_normalidad('IMC')

# Crear una matriz de correlación de Pearson
correlation_matrix_pearson = df.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_pearson, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black',  vmin=-1, vmax=1)
plt.title('Matriz de Correlación de Pearson')
plt.savefig('./img/matriz_correlacion_pearson.png', dpi=300, bbox_inches='tight')

# Matriz de correlación de Spearman
correlation_matrix_spearman = df.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_spearman, annot=True, cmap='YlGn', linewidths=0.5, linecolor='black', vmin=-1, vmax=1)
plt.title('Matriz de Correlación de Spearman')
plt.savefig('./img/matriz_correlacion_spearman.png', dpi=300, bbox_inches='tight')



# Matriz de regresión lineal de las variables Edad, IMC y Avg_Glucosa
sns.pairplot(df[['Edad', 'IMC', 'Avg_Glucosa']], kind='reg', plot_kws={'line_kws':{'color':'red', 'lw':1}, 'scatter_kws': {'alpha':0.8, 's':40, 'color':'gray'}})
plt.savefig('./img/matriz_regresion_lineal.png', dpi=300, bbox_inches='tight')

variables_discretas = ['Sex', 'Hipertension', 'Cardiopatia', 'Casado', 'Tipo_Trabajo', 'Tipo_Residencia', 'Fumar']

# duplas = combinations(variables_discretas, 2)

for pareja in variables_discretas:
    tabla_contingencia = pd.crosstab(df['Accidentes'], df[pareja])
    
    # # Crear un heatmap de la tabla de contingencia
    plt.figure(figsize=(10, 8))
    sns.heatmap(tabla_contingencia, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Tabla de Contingencia - Heatmap (Accidentes Cerebrovasculares vs {pareja})')
    plt.xlabel('Accidentes Cerebrovasculares')
    plt.ylabel(pareja)
    plt.savefig(f'./img/Heatmap/Accidentes_{pareja}.png', dpi=300, bbox_inches='tight')
    
    
    # Crear un gráfico de barras apiladas de la tabla de contingencia
    tabla_contingencia.plot(kind='bar', stacked=True, figsize=(10, 8))
    plt.title(f'Tabla de Contingencia - Gráfico de Barras Apiladas (Accidentes Cerebrovasculares vs {pareja})')
    plt.xlabel('Accidentes Cerebrovasculares')
    plt.ylabel('Frecuencia')
    plt.legend(title=pareja)
    plt.savefig(f'./img/Barplot/Accidentes_{pareja}.png', dpi=300, bbox_inches='tight')


plt.close()


def CrearCollage(ruta, nombre):
    # Directorio donde se guardan las imágenes de gráficos de barras apiladas
    barplot_dir = './img/Tablas/barplots/'
    barplot_images = [os.path.join(ruta, img) for img in os.listdir(ruta) if img.endswith('.png')]

    # Tamaño del collage
    collage_width = 1000
    collage_height = 2000

    # Crear una nueva imagen en blanco para el collage
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

    # Tamaño de cada imagen en el collage
    image_width = collage_width // 2
    image_height = collage_height // 4

    # Posición inicial
    x_offset = 0
    y_offset = 0

    # Añadir cada imagen al collage
    for img_path in barplot_images:
        img = Image.open(img_path)
        img = img.resize((image_width, image_height))
        collage.paste(img, (x_offset, y_offset))
        x_offset += image_width
        if x_offset >= collage_width:
            x_offset = 0
            y_offset += image_height

    # Guardar el collage
    collage.save(ruta +'/Collage/collage_' + nombre + '.png')


CrearCollage('./img/Barplot/', 'barplot')

CrearCollage('./img/Heatmap/', 'heatmap')