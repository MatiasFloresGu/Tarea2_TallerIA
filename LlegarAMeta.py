import numpy as np
import gym
from gym import spaces

class GridEnvironment(gym.Env):
    def __init__(self):
        super(GridEnvironment, self).__init__()
        self.grid_size = 5
        self.state = (0, 0)  # Estado inicial
        self.goal = (4, 4)  # Objetivo
        self.restricted_zones = [(2, 2), (3, 3), (3,4)]  # Zonas restringidas
        
        # Espacio de acciones: mover arriba, abajo, izquierda, derecha
        self.action_space = spaces.Discrete(4)
        # Espacio de observaciones: posición actual del agente
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)

    def step(self, action):
        x, y = self.state
        if action == 0 and y > 0:  # Mover arriba
            y -= 1
        elif action == 1 and y < self.grid_size - 1:  # Mover abajo
            y += 1
        elif action == 2 and x > 0:  # Mover izquierda
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # Mover derecha
            x += 1
        
        new_state = (x, y)
        reward = -1  # Penalización por cada paso
        
        # Comprobación de restricciones
        if new_state in self.restricted_zones:
            reward = -10  # Penalización alta por entrar en zonas restringidas
            new_state = self.state  # Mantener el estado anterior
        
        # Comprobación de objetivo alcanzado
        if new_state == self.goal:
            reward = 100
            done = True
        else:
            done = False
        
        self.state = new_state
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = (0, 0)
        return np.array(self.state)

    def render(self, mode="human"):
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                if (i, j) == self.state:
                    row += "A "  # Representa al agente
                elif (i, j) == self.goal:
                    row += "G "  # Representa el objetivo
                elif (i, j) in self.restricted_zones:
                    row += "X "  # Representa las zonas restringidas
                else:
                    row += ". "
            print(row)
        print()

# Ejemplo de uso
env = GridEnvironment()
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Seleccionar acción aleatoria
    state, reward, done, _ = env.step(action)
    env.render()

print("Episodio terminado.")