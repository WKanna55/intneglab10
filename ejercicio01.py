import numpy as np
import matplotlib.pyplot as plt
# Datos de entrenamiento
X = np.array([[3, 3],[4, 3],[1, 1],[1, -1],[2, -2],[3, -2]])
y = np.array([1, 1, 1, -1, -1, -1])  # Etiquetas: +1 o -1
# Inicialización de pesos y sesgo
w = np.zeros(X.shape[1])
b = 0
learning_rate = 0.1
epochs = 1000
# Entrenamiento (SVM muy simplificada)
for epoch in range(epochs):
    for i in range(len(X)):
        if y[i] * (np.dot(X[i], w) + b) < 1:
            w += learning_rate * (y[i] * X[i])
            b += learning_rate * y[i]
# Mostrar resultados
print("Pesos:", w)
print("Sesgo:", b)
# Visualización
# Crear figura
plt.figure(figsize=(8, 6))
# Puntos de clase +1
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Clase +1')
# Puntos de clase -1
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Clase -1')
# Generar línea del hiperplano: w1*x + w2*y + b = 0 → y = (-w1*x - b)/w2
x_vals = np.linspace(-1, 6, 100)
if w[1] != 0:
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k--', label='Hiperplano')
else:  # Si w[1] == 0, la línea sería vertical
    plt.axvline(x=-b / w[0], color='k', linestyle='--', label='Hiperplano')
# Opcional: márgenes (separación)
if w[1] != 0:
    margin = 1 / np.linalg.norm(w)
    y_vals_margin1 = -(w[0] * x_vals + b - 1) / w[1]
    y_vals_margin2 = -(w[0] * x_vals + b + 1) / w[1]
    plt.plot(x_vals, y_vals_margin1, 'g--', linewidth=0.8, label='Margen')
    plt.plot(x_vals, y_vals_margin2, 'g--', linewidth=0.8)
# Personalizar gráfico
plt.title("Clasificación con SVM (Simplificada)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.xlim(-1, 6)
plt.ylim(-3, 5)
plt.show()
