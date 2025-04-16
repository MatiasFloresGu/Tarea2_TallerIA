import gymnasium as gym
import numpy as np
import random

# Crear entorno con renderizado de texto
env = gym.make("Taxi-v3", render_mode="ansi")

# Inicializar tabla Q con ceros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Parámetros de entrenamiento optimizados
alpha = 0.2 
gamma = 0.95 
epsilon = 1.0 
epsilon_min = 0.1
epsilon_decay = 0.999
episodios = 5000
pasos_max = 200 

acciones = ["South", "North", "East", "West", "Pickup", "Dropoff"]

print("Iniciando entrenamiento...\n")
for ep in range(episodios):
    estado, _ = env.reset()
    terminado = False
    pasos = 0

    while not terminado and pasos < pasos_max:
        # Selección de acción basada en exploración-explotación
        if random.uniform(0, 1) < epsilon:
            accion = env.action_space.sample()  # Exploración aleatoria
        else:
            accion = np.argmax(q_table[estado])  # Explotación de valores Q aprendidos

        # Tomar acción y observar resultado
        nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)

        # Actualización de la tabla Q con una ecuación de Bellman mejorada
        q_table[estado, accion] = q_table[estado, accion] + alpha * (
            recompensa + gamma * np.max(q_table[nuevo_estado]) - q_table[estado, accion]
        )

        estado = nuevo_estado
        pasos += 1

    # Reducción progresiva de epsilon para mantener exploración por más tiempo
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (ep + 1) % 500 == 0:  # Ajustamos la frecuencia del mensaje de progreso
        print(f"Episodio {ep + 1} completado.")

print("\nEntrenamiento finalizado.\n")

# Evaluación del agente entrenado
estado, _ = env.reset()
terminado = False
pasos = 0
recompensa_total = 0

print("Simulación del agente entrenado:\n")
while not terminado and pasos < pasos_max:
    print(env.render())
    accion = np.argmax(q_table[estado])
    print(f"Acción elegida: {acciones[accion]}")
    estado, recompensa, terminado, truncado, _ = env.step(accion)
    recompensa_total += recompensa
    pasos += 1

print("\nFin de simulación.")
print(f"Pasos realizados: {pasos}")
print(f"Recompensa total: {recompensa_total}")