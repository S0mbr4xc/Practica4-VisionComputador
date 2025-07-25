import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Etiquetas ficticias
labels = ['circulo', 'triangulo', 'cuadrado']
y_true = (
    ['circulo'] * 10 +
    ['triangulo'] * 10 +
    ['cuadrado'] * 10
)
y_pred = [
    # Círculos: 9 correctos, 1 triángulo
    *(['circulo'] * 9 + ['triangulo']),
    # Triángulos: 9 correctos, 1 cuadrado
    *(['triangulo'] * 9 + ['cuadrado']),
    # Cuadrados: 8 correctos, 1 círculo, 1 triángulo
    *(['cuadrado'] * 8 + ['circulo', 'triangulo'])
]

# Cálculo de la matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Mostrar la matriz de confusión
disp.plot()
plt.title('Matriz de Confusión Ficticia')
plt.tight_layout()
plt.show()
