import gymnasium as gym
from gym import spaces
import numpy as np

# Definición de un entorno personalizado que hereda de gym.Env
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Espacio de observación: vector de 4 valores continuos entre 0 y 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # Espacio de acción: 3 acciones discretas posibles (0, 1, 2)
        self.action_space = spaces.Discrete(3)
        # Estado inicial aleatorio
        self.state = np.random.rand(4)
        # Número máximo de pasos por episodio
        self.steps_left = 100

    def reset(self):
        # Reinicia el entorno al estado inicial al comenzar un nuevo episodio
        self.state = np.random.rand(4)
        self.steps_left = 100
        return self.state

    def step(self, action):
        # Simula la toma de una acción en el entorno
        reward = np.random.rand()  # Recompensa aleatoria
        self.steps_left -= 1       # Se reduce el número de pasos restantes
        done = self.steps_left <= 0  # El episodio termina cuando se acaban los pasos
        self.state = np.random.rand(4)  # Se genera un nuevo estado aleatorio
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Muestra el estado actual del entorno
        print(f"Estado actual: {self.state}")

# Simulación del entorno durante 100 episodios
env = CustomEnv()
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # Selección de una acción aleatoria
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
    # Muestra la recompensa total obtenida en el episodio
    print(f"Recompensa total en episodio {episode + 1}: {total_reward}")
