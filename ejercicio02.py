import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Leer el archivo Excel
df = pd.read_excel("Data10.xlsx")

# 2. Seleccionar características y etiquetas
X = df[["Precio actual", "Precio final"]].values
y = df["Estado"].apply(lambda x: 1 if x == "Alto" else -1).values  # Alto = 1, Bajo = -1

# 3. Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo SVM lineal
modelo = svm.SVC(kernel="linear")
modelo.fit(X_train, y_train)

# 5. Predicción en datos de prueba
y_pred = modelo.predict(X_test)

# 6. Mostrar resultados en consola
print("=== Resultados del Modelo ===")
print("Pesos (coef_):", modelo.coef_)
print("Sesgo (intercept_):", modelo.intercept_)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Bajo", "Alto"]))

# 7. Visualización
plt.figure(figsize=(8, 6))

# Colorear por clase real
for clase, color, etiqueta in zip([1, -1], ['blue', 'red'], ['Alto', 'Bajo']):
    plt.scatter(
        X[y == clase][:, 0],
        X[y == clase][:, 1],
        c=color,
        label=etiqueta
    )

# Dibujar el hiperplano
w = modelo.coef_[0]
b = modelo.intercept_[0]
x_vals = np.linspace(min(X[:,0])-1, max(X[:,0])+1, 100)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, 'k--', label="Hiperplano SVM")

plt.xlabel("Precio actual")
plt.ylabel("Precio final")
plt.title("Clasificación SVM: Estado (Alto/Bajo)")
plt.legend()
plt.grid(True)
plt.show()
