import numpy as np
import random

#Llenado de matris
n_elementos = 5
max_peso = 20

matriz = []
for _ in range(n_elementos):
    valor = random.randint(1,5)
    peso = random.randint(1,10)
    matriz.append([valor, peso])

print("Elementos:")
for i, fila in enumerate(matriz):
    print(f"Elemento {i}: Valor= {fila[0]} / Peso= {fila[1]}")

#Parametros
alpha = 0.1# Tasa de aprendizaje: Controla cuánto se actualiza el valor Q en cada paso.
gamma = 0.9# Factor de descuento: Determina la importancia de las recompensas futuras.
epsilon = 0.1# Probabilidad de exploración: elegir una acción aleatoria en lugar de la óptima
repe = 10000 #repeticiones

Q = {}

#Entrenamiento del modelo
for episodio in range(repe):
    indice = 0
    peso_actual = 0

    while indice < n_elementos:
        estado = (indice, peso_actual)

        if estado not in Q:
            Q[estado] = [0, 0]  

        # Política greedy
        if random.random() < epsilon:
            accion = random.choice([0, 1])
        else:
            accion = np.argmax(Q[estado])

        valor, peso = matriz[indice]

        #Fase de recompensa
        if accion == 1 and peso_actual + peso <= max_peso:
            recompensa = valor
            nuevo_peso = peso_actual + peso
        elif accion == 1 and peso_actual + peso > max_peso:
            recompensa = -10  
            nuevo_peso = peso_actual
        else:
            recompensa = 0
            nuevo_peso = peso_actual

        siguiente_estado = (indice + 1, nuevo_peso)
        if siguiente_estado not in Q:
            Q[siguiente_estado] = [0, 0]

        # Actualización Q-learning
        Q[estado][accion] += alpha * (recompensa + gamma * max(Q[siguiente_estado]) - Q[estado][accion])

        indice += 1
        peso_actual = nuevo_peso

# Prueba real
indice = 0
mochila = []
valor_Total = 0
peso_Actual = 0

while indice < n_elementos:
    estado = (indice, peso_Actual)

    if estado in Q:
        accion = np.argmax(Q[estado])
    else:
        accion = 0  

    valor, peso = matriz[indice]

    if accion == 1 and peso_Actual + peso <= max_peso:
        mochila.append((indice, valor, peso))
        valor_Total += valor
        peso_Actual += peso

    indice += 1

#Resultados
print(f"\nPeso Maximo:{max_peso}")
print("\nMejor combinación encontrada:")
for indice_E, valor_M, peso_M in mochila:
    print(f"Elemento {indice_E}: Valor= {valor_M} / Peso= {peso_M}")

print(f"\nValor total= {valor_Total}")
print(f"Peso total= {peso_Actual}")
