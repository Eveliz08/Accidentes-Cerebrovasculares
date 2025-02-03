import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import kstest, shapiro, anderson

import seaborn as sns
from scipy.stats import norm
from scipy.stats import pearsonr

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
tablaEstCentro_table.scale(1.2, 1.5) 

# Se le da un color gris de fondo a la cabecera de las filas y las columnas
for key, cell in tablaEstCentro_table.get_celld().items():
    if key[0] == 0 or key[1] == -1:
        cell.set_facecolor('lightgray')

plt.savefig('./img/Estadísticos_de_centro.png', dpi=300, bbox_inches='tight')


# Guarda la tabla de los estadísticos de dispersión como imágenes
fig, ax = plt.subplots(figsize=(12, 2)) 
ax.axis('tight')
ax.axis('off')
tablaEstDispersion_table = ax.table(cellText=tablaEstDispersion.values, colLabels=tablaEstDispersion.columns, rowLabels=tablaEstDispersion.index, cellLoc='center', loc='center')
tablaEstDispersion_table.auto_set_font_size(False)
tablaEstDispersion_table.set_fontsize(10)
tablaEstDispersion_table.scale(1.2, 1.5)  # Increase the width and height of the table

# Se le da un color gris de fondo a la cabecera de las filas y las columnas
for key, cell in tablaEstDispersion_table.get_celld().items():
    if key[0] == 0 or key[1] == -1:
        cell.set_facecolor('lightgray')

plt.savefig('./img/Estadísticos_de_dispersión.png', dpi=300, bbox_inches='tight')


#Crear gráficos pastel de las variables cualitativas

#Se crea un método que hace el gráfico pastel de las variables bernulli del dataset
def crear_grafico_pastel(variable, label1, label2):
    df_map = df[variable].map({0: label1, 1: label2})
    tabla = pd.crosstab(index = df_map, columns='Frecuencia')
    
    plt.figure(figsize=(6,6))
    colors = sns.color_palette('pastel')[0:2]
    plt.pie(tabla['Frecuencia'], labels=[label1, label2], autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    plt.title("Gráfico de Pastel de la variable " + variable)

    plt.savefig("./img/grafico_pastel_" + variable + ".png", dpi = 300)

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

    plt.savefig("./img/Histograma_" + variable + ".png", dpi = 300)


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

    plt.savefig("./img/Boxplt_" + variable + ".png", dpi = 300)

Crear_Boxplot('Edad', 'blue')
Crear_Boxplot('IMC', 'red')
Crear_Boxplot('Avg_Glucosa', 'yellow')









#Pruebas de normalidad para evaluar si las variables cuantitativas siguen una distribucion normal
print('Variable Edad')
#Prueba para la variable Edad
mean = df['Edad'].mean()
std = df['Edad'].std()

# Prueba de Kolmogorov-Smirnov para normalidad
stat, p_value = kstest(df['Edad'], 'norm', args=(mean, std))
print(f'Kolmogorov-Smirnov test - Estadístico: {stat:.4f}, p-valor: {p_value:.4f}')

# Prueba de Shapiro-Wilk para normalidad
stat, p_value = shapiro(df['Edad'])
print(f'Shapiro-Wilk test - Estadístico: {stat:.4f}, p-valor: {p_value:.4f}')

# Prueba de Anderson-Darling para normalidad
result = anderson(df['Edad'])
print(f'Anderson-Darling test - Estadístico: {result.statistic:.4f}')
for i, critical in enumerate(result.critical_values):
    signif_level = result.significance_level[i]
    print(f'Nivel de significancia {signif_level}%: Valor crítico {critical}')
    if result.statistic < critical:
        print(f"Los datos parecen seguir una distribución normal al {signif_level}% de nivel de significancia.")
    else:
        print(f"Los datos no siguen una distribución normal al {signif_level}% de nivel de significancia.")
        

print()
print('Variable Avg_Glucosa')
#Prueba para la variable Avg_Glucosa
mean = df['Avg_Glucosa'].mean()
std = df['Avg_Glucosa'].std()

# Prueba de Kolmogorov-Smirnov para normalidad
stat, p_value = kstest(df['Avg_Glucosa'], 'norm', args=(mean, std))
print(f'Kolmogorov-Smirnov test - Estadístico: {stat:.4f}, p-valor: {p_value:.4f}')

# Prueba de Shapiro-Wilk para normalidad
stat, p_value = shapiro(df['Avg_Glucosa'])
print(f'Shapiro-Wilk test - Estadístico: {stat:.4f}, p-valor: {p_value:.4f}')

# Prueba de Anderson-Darling para normalidad
result = anderson(df['Avg_Glucosa'])
print(f'Anderson-Darling test - Estadístico: {result.statistic:.4f}')
for i, critical in enumerate(result.critical_values):
    signif_level = result.significance_level[i]
    print(f'Nivel de significancia {signif_level}%: Valor crítico {critical}')
    if result.statistic < critical:
        print(f"Los datos parecen seguir una distribución normal al {signif_level}% de nivel de significancia.")
    else:
        print(f"Los datos no siguen una distribución normal al {signif_level}% de nivel de significancia.")



print()
print('Variable IMC')
#Prueba para la variable IMC
mean = df['IMC'].mean()
std = df['IMC'].std()

# Prueba de Kolmogorov-Smirnov para normalidad
stat, p_value = kstest(df['IMC'], 'norm', args=(mean, std))
print(f'Kolmogorov-Smirnov test - Estadístico: {stat:.4f}, p-valor: {p_value:.4f}')

# Prueba de Shapiro-Wilk para normalidad
stat, p_value = shapiro(df['IMC'])
print(f'Shapiro-Wilk test - Estadístico: {stat:.4f}, p-valor: {p_value:.4f}')

# Prueba de Anderson-Darling para normalidad
result = anderson(df['IMC'])
print(f'Anderson-Darling test - Estadístico: {result.statistic:.4f}')
for i, critical in enumerate(result.critical_values):
    signif_level = result.significance_level[i]
    print(f'Nivel de significancia {signif_level}%: Valor crítico {critical}')
    if result.statistic < critical:
        print(f"Los datos parecen seguir una distribución normal al {signif_level}% de nivel de significancia.")
    else:
        print(f"Los datos no siguen una distribución normal al {signif_level}% de nivel de significancia.")


# Crear una matriz de correlación de las variables del DataFrame
correlation_matrix = df.corr()

# Crear un heatmap de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Matriz de Correlación de las Variables')
plt.savefig('./img/matriz_correlacion.png', dpi=300, bbox_inches='tight')

# Crear gráficos de dispersión para todas las combinaciones posibles de las variables
variables = ['Edad', 'IMC', 'Avg_Glucosa']
sns.pairplot(df[variables], diag_kind='kde', plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'})
plt.savefig('./img/graficos_dispersion.png', dpi=300, bbox_inches='tight')
