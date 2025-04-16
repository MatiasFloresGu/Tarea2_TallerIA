import numpy as np
import random

# Mapa original con A y B
original_map = [
    [5, 3, 'A', 2, 8, 1, 6, 4, 9, 2],
    [3, 8, 4, 1, 9, 5, 2, 7, 6, 3],
    [6, 1, 9, 7, 3, 4, 8, 5, 2, 1],
    [2, 5, 8, 3, 6, 9, 1, 4, 7, 5],
    [4, 7, 1, 5, 2, 8, 3, 9, 6, 4],
    [9, 2, 6, 4, 1, 7, 5, 3, 8, 2],
    [1, 6, 3, 9, 5, 2, 7, 8, 4, 1],
    [7, 4, 2, 8, 6, 3, 9, 1, 5, 7],
    [8, 9, 5, 6, 4, 1, 2, 3, 7, 8],
    [7, 3, 7, 2, 9, 5, 4, 6, 1, 'B']
]

# Convertimos A y B a valores numéricos
map_grid = []
start_pos = None
end_pos = None

for i, row in enumerate(original_map):
    map_row = []
    for j, val in enumerate(row):
        if val == 'A':
            start_pos = (i, j)
            map_row.append(0)  # Recompensa de inicio
        elif val == 'B':
            end_pos = (i, j)
            map_row.append(100)  # Gran recompensa por llegar
        else:
            map_row.append(val)
    map_grid.append(map_row)

map_grid = np.array(map_grid)

# Parámetros Q-learning
rows, cols = map_grid.shape
q_table = np.zeros((rows, cols, 4))  # 4 acciones: 0=up, 1=down, 2=left, 3=right
learning_rate = 0.1
discount = 0.9
epsilon = 0.1  # exploración

# Movimientos: (dx, dy)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

def is_valid(x, y):
    return 0 <= x < rows and 0 <= y < cols

# Entrenamiento
for episode in range(1000):
    x, y = start_pos
    while (x, y) != end_pos:
        # Elegir acción
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  # Explorar
        else:
            action = np.argmax(q_table[x, y])  # Explotar

        dx, dy = actions[action]
        nx, ny = x + dx, y + dy

        if not is_valid(nx, ny):
            continue  # No salir del mapa

        reward = map_grid[nx, ny]
        next_max = np.max(q_table[nx, ny])
        # Actualización Q
        q_table[x, y, action] += learning_rate * (reward + discount * next_max - q_table[x, y, action])

        x, y = nx, ny

# Seguimos la política aprendida desde A hasta B
path = [start_pos]
x, y = start_pos
while (x, y) != end_pos:
    action = np.argmax(q_table[x, y])
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy

    if not is_valid(nx, ny):
        break  # evitar loops infinitos en errores

    path.append((nx, ny))
    x, y = nx, ny

# Imprimimos el camino y el total de puntos acumulados
total_score = sum(map_grid[x][y] for x, y in path)
print("Camino óptimo:", path)
print("Score total acumulado:", total_score)
