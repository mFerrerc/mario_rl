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

    # Primero, importamos las librerías necesarias para el vectorizado de entornos
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env

    # Vamos a utilizar la función make_vec_env para crear un entorno vectorizado.
    # Esta función crea un entorno vectorizado utilizando el entorno base que hemos creado anteriormente.
    # El número de entornos vectorizados por defect es 4, pero cambiadlo en base a vuestra CPU y analisis de rendimiento.

    ### CAMBIA ###

    from stable_baselines3.common.monitor import Monitor
    import random

    REPLICAS_PER_LEVEL = 4
    NUM_ENVS = len(LEVEL_IDS)*REPLICAS_PER_LEVEL # aquí n_envs
    BASE_SEED   = 33

    # 2) Creamos una "fábrica" que:
    #    - construya el env con create_mario_env(world,stage,…)
    #    - lo envuelva en Monitor (para monitor_dir)
    #    - lo aisle con su propia semilla (seed + rank)
    def make_level_env_fn(level_tuple, rank):
        world, stage = level_tuple
        def _init():
            # 1) Crea el env “crudo” con todos tus wrappers
            env = create_mario_env(
                world, stage,
                action_type=ACTION_TYPE,
                n_frames_repeat=LEVEL_ACTION_REPEAT[(world, stage)],
                n_frames_stack=N_FRAMES_STACK,
                render_mode="rgb_array"
            )
            # 2) Monitor para volcar reward/length por episodio
            env = Monitor(
                env,
                filename=f"./mario_monitor_dir/_robusto_monitor_{world}_{stage}_{rank}.csv",
                allow_early_resets=True
            )

            # 3) Gymnasium: resetea con semilla en vez de env.seed()
            _ , _info = env.reset(seed=BASE_SEED + rank)

            return env
        return _init

    # 3) Montamos la lista de factories
    env_fns = []
    rank = 0
    for level in LEVEL_IDS:
        for _ in range(REPLICAS_PER_LEVEL):
            env_fns.append(make_level_env_fn(level, rank))
            rank += 1

    # 4) Instanciamos SubprocVecEnv con todas ellas
    #    Aquí entra vec_env_cls=SubprocVecEnv y n_envs = len(env_fns)
    env = SubprocVecEnv(env_fns, start_method="spawn")

    print(f"→ Creamos {NUM_ENVS} envs: {len(LEVEL_IDS)} niveles × {REPLICAS_PER_LEVEL} réplicas")

            ###########################################################################
            ###########################################################################
            ###########################################################################


    from stable_baselines3.common.callbacks import BaseCallback
    import os

    # Callback para guardar el modelo y registrar la información durante el entrenamiento.
    # Este callback se ejecuta cada cierto número de pasos y guarda el modelo en la ruta especificada.
    class TrainAndLogCallback(BaseCallback):
        def __init__(self, name, check_freq, save_path, start_steps=0, verbose=1):
            super(TrainAndLogCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path
            self.start_steps = start_steps
            self.name = name

        def _init_callback(self):
            # Creamos la carpeta de guardado si no existe.
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            # Guardamos el modelo cada check_freq pasos.
            if self.n_calls % self.check_freq == 0:
                if self.save_path is not None:
                    # convierto a pasos reales
                    real_steps = self.n_calls * self.model.n_envs
                    filename   = os.path.join(self.save_path, f"model_{real_steps}_{self.name}")
                    self.model.save(filename)
            return True

    #TODO: (Opcional) Implementad otros callbacks si lo considerais necesario.

    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList

    # 2) Callback de parada temprana al alcanzar un umbral de recompensa media
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=350,    # adapta este umbral a vuestra métrica
        verbose=1
    )

    # 1) Callback de evaluación periódica en un entorno aparte

    BASE_SEED = 33

    # 1) fábrica de entornos de evaluación
    def make_eval_env_fn(level_tuple, rank):
        world, stage = level_tuple
        def _init():
            env = create_mario_env(
                world, stage,
                action_type=ACTION_TYPE,
                n_frames_repeat=LEVEL_ACTION_REPEAT[(world, stage)],
                n_frames_stack=N_FRAMES_STACK,
                render_mode="rgb_array"         # para que .render() muestre ventana
            )
            # opcional: monitor para cada nivel
            '''env = Monitor(env,
                        filename=f"./eval_monitor/robusto_level{world}_{stage}.csv",
                        allow_early_resets=True)'''
            # .reset ahora con semilla
            env.reset(seed=BASE_SEED + rank)
            return env
        return _init

    # 2) instanciamos uno por cada LEVEL_IDS
    eval_fns = []
    for idx, lvl in enumerate(LEVEL_IDS):
        eval_fns.append(make_eval_env_fn(lvl, idx))
    # 3) vectorizamos
    eval_env = SubprocVecEnv(eval_fns, start_method="spawn")   

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_robust/",
        log_path="./eval_logs/",
        eval_freq=10_000/NUM_ENVS,          
        n_eval_episodes=1*len(LEVEL_IDS),
        deterministic=True,
        render=True,
        #callback_on_new_best=stop_callback
    )

    if ACTION_TYPE == SIMPLE_MOVEMENT:
        run_id = f"robusto_w{WORLD}_ac-SIMPLE_MOVEMENT-"
    elif ACTION_TYPE == RIGHT_ONLY:
        run_id = f"robusto_w{WORLD}__ac-RIGHT_ONLY-" 

    save_path = os.path.join("./mario_models", run_id)

    callback = CallbackList([ 
        TrainAndLogCallback(name=run_id, check_freq=CHECK_FREQ//NUM_ENVS, save_path=save_path, start_steps=0, verbose=1),
        eval_callback, 
        #stop_callback, 
        ])

            ###########################################################################
            ###########################################################################
            ###########################################################################

    #TODO: Inicializa el modelo con el algoritmo deseado (PPO, DQN, A2C, etc.), el entorno vectorizado y los parámetros deseados. 
    # Aseguraos de indicar verbose=1 para que se vea el progreso del entrenamiento.

    from stable_baselines3 import PPO
    import os
    from stable_baselines3.common.utils import get_linear_fn

    log_path = os.path.join("./train", run_id)

    total_timesteps = 2_000_000*NUM_ENVS

    model = PPO(
        "CnnPolicy",           # arquitectura CNN para procesar los stacks de frames
        env,                    # entorno vectorizado creado arriba
        learning_rate=2.5e-4,   # LR típico para PPO
        n_steps=128,            # número de timesteps por rollout
        batch_size=64,          # tamaño de minibatch
        n_epochs=8,             # pasadas de SGD por rollout
        gamma=0.95,             # factor de descuento
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs={'normalize_images': False},
        device='cuda:0'          # <- Le dices que use la GPU 0
    )

    # TODO: Entrena el modelo con el número de pasos deseado
    # Para un entrenamiento rápido pero efectivo, vamos a usar 1e6 pasos. (approx. 1h en una CPU i7 segun el vectorizado que useis)
    # Este número puede ser ajustado en base a la velocidad de entrenamiento y el rendimiento del modelo.
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    # Código de entrenamiento aquí!

    # Guardamos el modelo final.
    #model.save("mario_final_model")
    # Cerramos el entorno.
    env.close()