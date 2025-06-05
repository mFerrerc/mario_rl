import gymnasium as gym
# Gymnasium ofrece diferentes tipos de espacios para definir el espacio de acción y el espacio de observación.
# En este caso, usamos Box para definir un espacio de observación continuo, y Discrete para un espacio de acción discreto.
from gymnasium.spaces import Box, Discrete

# Para modificar el entorno de Gymnasium, usamos la clase Wrapper.
# Wrapper es una clase base que permite modificar el comportamiento de un entorno sin cambiar su implementación.
from gymnasium import Wrapper

# Del emulador de NES, importamos el espacio de acción que emula el joystick.
from nes_py.wrappers import JoypadSpace

# Importamos el entorno de Super Mario Bros.
import gym_super_mario_bros
# Como indica la documentacion, el entorno de Super Mario Bros. tiene diferentes espacios de acción.
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

    ###########################################################################
    ###########################################################################
    ########################################################################### 

import cv2
import numpy as np

# Dimensiones de la imagen procesada
HEIGHT = 84
WIDTH = 84

# Aplica un procesamiento a la imagen del entorno para que sea más fácil de manejar.
def process_frame(frame):    
    if frame is not None:        
        #TODO: Convierte el frame a escala de grises
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #TODO: Redimensiona la imagen
        resized = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        #TODO: Normaliza la imagen dividiendo por 255.0
        normalized = resized / 255.0

        return normalized.astype(np.float32)  # Ensure the output is float32
    else:
        return np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    ###########################################################################
    ###########################################################################
    ###########################################################################
 
# Creamos una clase que hereda de Wrapper para modificar el entorno.
# Esta clase se encargará de procesar la imagen y modificar la recompensa.
class CustomReward(Wrapper):
    def __init__(self, env=None, w_kill=1.0):
        super(CustomReward, self).__init__(env)
        # Actualizamos el espacio de observación y el espacio de acción.
        self.observation_space = Box(0.0, 1.0, shape=(HEIGHT, WIDTH), dtype=np.float32)
        self.action_space = Discrete(env.action_space.n)
        # Inicializamos la información adicional.
        self.info = {}
        # Inicializamos la posición x inicial de Mario.
        self.current_x = 40

        #TODO: Inicializad las variables necesarias para vuestra implementación.

        self.prev_coins  = 0            # monedas recogidas en el paso anterior
        self.w_kill = w_kill
        self.flag_rewarded = False

    def step(self, action):
        ### Modificamos la función step() para procesar la imagen y modificar la recompensa. ###

        # Ejecutamos la acción en el entorno y obtenemos el nuevo estado, la recompensa, si ha terminado el episodio y la información adicional.
        obs, reward, done, truncated, info = self.env.step(action)
        self.info = info
        #TODO: Aplicamos el procesamiento a la imagen.
        obs = process_frame(obs)

        #TODO: Personalizamos la recompensa.
        #TODO: Vamos a añadir una recompensa terminal positiva si Mario llega a la meta y una negativa si no
        if info.get("flag_get", False) and not self.flag_rewarded:
            reward += 50.0
            self.flag_rewarded = True
            print("🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩🚩")
        elif done and not info.get("flag_get", False):
            reward -= 20.0

        # 2. Monedas
        coin_diff = info["coins"] - self.prev_coins
        reward += 1 * coin_diff
        self.prev_coins = info["coins"]

        '''# + Recompensa por muertes de enemigos
        reward += 0.05 * (info["score"] - self.prev_score)
        self.prev_score = info["score"]

        5. Potential-based shaping
        reward += 0.1 * (info["x_pos"] - self.last_x)
        self.last_x = info["x_pos"]'''
        
        reward += 0.05 * (info["y_pos"] - self.prev_y)   # subir = +, caer = –
        self.prev_y = info["y_pos"]

        return obs, reward / 10., done, truncated, info

    def reset(self, **kwargs):
        ### Modificamos la función reset() para procesar la imagen y reiniciar las variables necesarias. ###
        #TODO: Reiniciad las variables necesarias para vuestra implementación.
        kwargs.pop('seed', None)

        # Reiniciamos el entorno y obtenemos el estado inicial procesado y la información adicional.
        obs, info = self.env.reset(**kwargs)

        # AÑADIDAS
        self.prev_coins    = 0
        self.flag_rewarded = False
        self.prev_score    = info.get("score", 0)
        self.last_x     = info.get("x_pos", 0)
        self.prev_y = info.get("y_pos", 0)

        return process_frame(obs), info
    
    ###########################################################################
    ###########################################################################
    ###########################################################################

from collections import deque
import numpy as np

# Wrapper para apilar frames del entorno.
# Esta clase se encargará de apilar los frames del entorno para crear un stack de frames.
# Esto es útil para que el agente pueda ver el movimiento de Mario y aprender a jugar mejor.
N_FRAMES = 4
class CustomStackFrames(Wrapper):
    def __init__(self, env, n_frames=N_FRAMES):
        super(CustomStackFrames, self).__init__(env)
        self._n_frames = n_frames
        # TODO: Actualiza el espacio de observaciones del entorno para que tenga en cuenta el número de frames apilados.
        
        self.observation_space = Box(
            low=0, high=1.0,
            shape=(self._n_frames, HEIGHT, WIDTH),
            dtype=np.float32
        )

        # TODO: Inicializa el stack de frames.
        self.frames = deque(maxlen=self._n_frames)

    def step(self, action):
        #TODO: Modifica la función step() para apilar los frames. Piensa qué hacer con la recompensa, y como manejar el final del episodio.
        '''done = False
        truncated = False'''

        # Ejecuta la acción en el entorno original
        obs_raw, reward, done, truncated, info = self.env.step(action)

        # Procesamos el nuevo frame y lo añadimos al stack
        frame = process_frame(obs_raw)
        self.frames.append(frame)

        # Construimos la observación apilada
        stacked_obs = np.stack(self.frames, axis=0)

        # Si el episodio termina o se trunca, limpiamos el stack para el próximo reset
        if done or truncated:
            self.frames.clear()

        # TODO: Acuerdate de devolver lo que devuelve la función step() original!

        # Devolvemos la observación apilada, recompensa, flags y info
        return stacked_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        # TODO: Modifica la función reset() para apilar los frames.
        # En este caso, considera que el stack de frames es el estado inicial repetido n veces.
        
        # Descartamos el seed (DummyVecEnv/VecEnv se lo pasa, pero el env legacy no lo acepta)
        kwargs.pop('seed', None)

        # Reiniciamos el entorno original
        obs_raw, info = self.env.reset(**kwargs)

        # Creamos el frame procesado y llenamos el stack con N copias
        frame = process_frame(obs_raw)
        self.frames = deque([frame] * self._n_frames, maxlen=self._n_frames)

        # Devolvemos la observación apilada inicial y la info
        return np.stack(self.frames, axis=0), info
    
########################## Nuevo WRAPPER ##########################

class ActionRepeat(Wrapper):
    """
    Wrapper que repite la misma acción durante `repeat` pasos consecutivos.
    - Acumula la recompensa.
    - Si `done` o `truncated` ocurren, deja de repetir y retorna inmediatamente.
    """
    def __init__(self, env, repeat: int = 4):
        super().__init__(env)
        self.repeat = repeat
        # Conservamos los espacios originales
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        # Solo delegamos al reset original
        return self.env.reset(**kwargs)

        ###########################################################################
        ###########################################################################
        ###########################################################################

def create_mario_env(world, stage, action_type, n_frames_repeat, n_frames_stack, render_mode):    
    #TODO: Crea el entorno base de Super Mario Bros. con el mundo y el nivel especificados.
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v0",
        render_mode=render_mode,
        apply_api_compatibility=True
    )

    # Envuelve el entorno en el wrapper del Joystick de NES para poder elegir las acciones.
    env = JoypadSpace(env, action_type)
    env = ActionRepeat(env, repeat=n_frames_repeat)
    #TODO: Envuelve el entorno en los wrappers de CustomStackFrames y CustomReward.
    env = CustomReward(env)
    env = CustomStackFrames(env, n_frames_stack)

    return env

        ###########################################################################
        ###########################################################################
        ###########################################################################

# Variables de configuración
# Puedes cambiar el mundo y el nivel para probar diferentes niveles de Super Mario Bros.
# También puedes cambiar el tipo de acción y el número de frames apilados.
ACTION_TYPE = SIMPLE_MOVEMENT
N_FRAMES_STACK = 4
# N_FRAMES_REPEAT = 4
CHECK_FREQ = 10_000

WORLD = 1

LEVEL_IDS = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    # …añade los que necesites
]

LEVEL_ACTION_REPEAT = {
    (1, 1): 4,
    (1, 2): 8,
    (1, 3): 2,   # ← control fino
    (1, 4): 4    # ← timing vs. Bowser
}

        ###########################################################################
        ###########################################################################
        ###########################################################################

if __name__ == "__main__":

    WORLD = 1
    STAGE = 3 # Cambiar para probar niveles
    

    # SB3 proporciona una función para evaluar el rendimiento del agente entrenado.
    # Esta función evalúa el rendimiento del agente en el entorno especificado y devuelve la recompensa media y la desviación estándar.
    from stable_baselines3.common.evaluation import evaluate_policy
    import time

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    ######### EVALUACION CUANTITATIVA #########
    # TODO: Creamos un nuevo entorno para evaluar el agente entrenado.
    # Este entorno es el mismo que el utilizado para entrenar el agente, pero sin el vectorizado. Vigilad tambien el tipo de renderizado!
    env = create_mario_env(
        WORLD, STAGE, action_type=ACTION_TYPE, n_frames_repeat=LEVEL_ACTION_REPEAT[(WORLD, STAGE)], n_frames_stack=N_FRAMES_STACK, render_mode="human"
    )
    env = DummyVecEnv([lambda: env])
    
    # 2) Entorno “crudo” para renderizar todos los frames
    raw_env = gym_super_mario_bros.make(
        f"SuperMarioBros-{WORLD}-{STAGE}-v0",
        render_mode="human",
        apply_api_compatibility=True
    )
    raw_env = JoypadSpace(raw_env, ACTION_TYPE)
    
    # TODO: Cargamos el modelo entrenado.
    
    model = PPO.load("./best_model_robust/best_model")
    
    #TODO: Evaluamos el rendimiento del agente en el entorno especificado y mostramos la recompensa media y la desviación estándar.
    N_EVAL_EPISODES = 1

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=N_EVAL_EPISODES, render=False
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    ######### EVALUACION CUALITATIVA #########
    # Importante! Los wrappers VecEnv devuelven solo done para el fin de estado, no trunk. 
    # Del mismo modo, el reset() devuelve solo el estado y no la info.
    # Tenlo en cuenta para el bucle de evaluación.

    #TODO: Implementa un bucle de evaluación para analizar el rendimiento del agente entrenado de forma cualitativa.
    # Consejo: añade time.sleep(0.02) tras renderizar el entorno para que la velocidad de juego sea más lenta y puedas ver mejor el rendimiento del agente.

    # Reset the environment
    obs = env.reset()
    raw_obs, _ = raw_env.reset()
    done = False

    try:
        for e in range(N_EVAL_EPISODES):
            while not done:
                # Obten la accion del modelo
                action, _ = model.predict(obs, deterministic=True)
                # Ejecuta la accion en el entorno y observa el nuevo estado, la recompensa, si ha terminado el episodio y la información adicional.
                obs, reward, done, info = env.step(action)
                # Renderiza el entorno para visualizar el juego.
                '''env.render()
                time.sleep(0.2)
                if done:
                    print("Episode finished. Resetting environment.")
                    obs = env.reset()'''

                # 3.3) En el env “raw” repetimos la misma acción N veces, renderizando cada frame
                for _ in range(LEVEL_ACTION_REPEAT[(WORLD, STAGE)]):
                    raw_obs, _, raw_done, _, _ = raw_env.step(int(action))
                    raw_env.render()         # aquí vemos cada frame
                    time.sleep(1 / 60)       # ralentiza a ~60 FPS
                    if raw_done:
                        print("Episode finished. Resetting environment.")
                        done = True
                        obs = env.reset()
                        raw_obs, _ = raw_env.reset()
                        break
            
            done = False
    except KeyboardInterrupt:
        print("Exiting...")
        env.close()
        raw_env.close()
        exit()
    # Close the environment
    env.close()
    raw_env.close()