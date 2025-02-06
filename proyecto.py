# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.feature_selection import chi2

# df = pd.read_csv('Cerebrovasculares.csv')

# # Seleccionar las variables independientes y la variable dependiente
# X = df[['Avg_Glucosa']]
# y = df['Accidentes']


# # Dividir el dataset en conjunto de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Crear el modelo de regresión logística
# model = LogisticRegression()
# # Entrenar el modelo
# model.fit(X_train, y_train)
# # Realizar predicciones
# y_pred = model.predict(X_test)

# # Evaluar el modelo
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print(f'Accuracy: {accuracy}')
# print('Confusion Matrix:')
# print(conf_matrix)
# print('Classification Report:')
# print(class_report)

# # Obtener los coeficientes del modelo
# coeficientes = pd.DataFrame({
#     'Variable': X.columns,
#     'Coeficiente': model.coef_[0]
# })
# print("Coeficientes del modelo:\n", coeficientes)


# # Generar el informe
# with open('informe.txt', 'w') as f:
#     f.write(f'Accuracy: {accuracy}\n')
#     f.write('Confusion Matrix:\n')
#     f.write(f'{conf_matrix}\n')
#     f.write('Classification Report:\n')
#     f.write(f'{class_report}\n')
#     f.write('Coeficientes del modelo:\n')
#     f.write(f'{coeficientes}\n')

