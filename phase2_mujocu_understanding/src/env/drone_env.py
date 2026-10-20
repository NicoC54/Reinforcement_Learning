import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class DroneLandingEnv(gym.Env):
    """
    Environnement Gymnasium pour l'atterrissage d'un drone sur une plateforme mobile
    avec trajectoire sinusoïdale (slalom et variations de vitesse).
    """
    def __init__(self):
        super().__init__()
        
        # 1. Chargement de la physique
        dossier_env = os.path.dirname(os.path.abspath(__file__))      # C'est le dossier 'env'
        dossier_src = os.path.dirname(dossier_env)                    # On remonte à 'src'
        chemin_xml = os.path.join(dossier_src, "xmls", "final.xml")
        
        self.model = mujoco.MjModel.from_xml_path(chemin_xml)
        self.data = mujoco.MjData(self.model)
        
        # 2. Espaces d'Action et d'Observation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Initialisation des variables internes
        self.vitesse_car_x = 0.0
        self.vitesse_car_y = 0.0
        self.ancienne_distance = 0.0
        self.current_step = 0
        self.last_action = np.zeros(4, dtype=np.float32)
        
        # Paramètres de la trajectoire sinusoïdale
        self.base_vel_x = 0.0
        self.amp_x = 0.0
        self.amp_y = 0.0
        self.freq_x = 0.0
        self.freq_y = 0.0
        self.phase_y = 0.0
        
        self.metadata = {"render_modes": ["human"]}
        self.viewer = None

    def _get_obs(self):
        # Positions
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        pos_relative = pos_drone - pos_car
        
        # Vitesses
        vel_lin_drone = self.data.joint("drone_root").qvel[0:3]
        vel_ang_drone = self.data.joint("drone_root").qvel[3:6]
        
        # Vitesse réelle de la plateforme (X et Y)
        vel_x_car = self.data.joint("Tx_car").qvel[0]
        vel_y_car = self.data.joint("Ty_car").qvel[0]
        vel_car = np.array([vel_x_car, vel_y_car, 0.0])
        
        # Vitesse RELATIVE (Drone - Cible)
        vel_relative = vel_lin_drone - vel_car
        
        # Orientation (Quaternion)
        quat_drone = self.data.joint("drone_root").qpos[3:7]

        obs = np.concatenate([
            pos_relative / 5.0,
            vel_relative / 5.0,
            quat_drone,
            vel_ang_drone / 10.0
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Randomisation du drone (Position, inclinaison et vitesse initiale)
        bruit_x = np.random.uniform(-2.0, 2.0)
        bruit_y = np.random.uniform(-2.0, 2.0)
        altitude = np.random.uniform(1.5, 3.5)
        
        self.data.joint("drone_root").qpos[0:3] = [-1.5 + bruit_x, 0.0 + bruit_y, altitude]
        self.data.joint("drone_root").qpos[3:7] = [1.0, np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0.0]
        self.data.joint("drone_root").qvel[0:3] = np.random.uniform(-0.3, 0.3, 3)
        
        # 2. Reset de la plateforme (Remise à zéro de la position)
        self.data.joint("Tx_car").qpos[0] = 0.0
        self.data.joint("Ty_car").qpos[0] = 0.0
        
        # 3. Paramètres de la trajectoire sinusoïdale pour cet épisode
        self.base_vel_x = np.random.uniform(0.2, 0.7)       # Vitesse d'avancée moyenne
        self.amp_x = np.random.uniform(0.1, 0.3)            # Variations d'accélération/freinage
        self.amp_y = np.random.uniform(0.2, 0.5)            # Amplitude du slalom latéral
        self.freq_x = np.random.uniform(0.5, 1.5)           # Fréquence des accélérations
        self.freq_y = np.random.uniform(0.5, 2.0)           # Fréquence du slalom
        self.phase_y = np.random.uniform(0, 2 * np.pi)      # Déphasage de départ
        
        mujoco.mj_forward(self.model, self.data)
        
        # Distance initiale pour le progrès et reset de la dernière action
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        self.ancienne_distance = float(np.linalg.norm(pos_drone - pos_car))
        self.last_action = np.zeros(4, dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Mapping [-1, 1] vers [0, 1]
        moteur_cmd = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        
        self.data.actuator("thrust_fl").ctrl[0] = moteur_cmd[0]
        self.data.actuator("thrust_fr").ctrl[0] = moteur_cmd[1]
        self.data.actuator("thrust_bl").ctrl[0] = moteur_cmd[2]
        self.data.actuator("thrust_br").ctrl[0] = moteur_cmd[3]
        
        # Calcul du temps écoulé (dt = 0.01s par step de l'agent)
        t = self.current_step * 0.01
        
        # Mise à jour des vitesses instantanées de la plateforme (Trajectoire dynamique)
        self.vitesse_car_x = self.base_vel_x + self.amp_x * np.sin(self.freq_x * t)
        self.vitesse_car_y = self.amp_y * np.cos(self.freq_y * t + self.phase_y)
        
        # Frame skip (5 sous-pas) : Application de la vitesse
        for _ in range(5):
            self.data.joint("Tx_car").qvel[0] = self.vitesse_car_x
            self.data.joint("Ty_car").qvel[0] = self.vitesse_car_y
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # Extraction des positions et distances
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        
        dist_xy = np.linalg.norm(pos_drone[:2] - pos_car[:2])
        dist_z = abs(pos_drone[2] - pos_car[2])
        vraie_distance = np.linalg.norm(pos_drone - pos_car)
        
        # Vitesses relatives et absolues
        vel_lin_drone = self.data.joint("drone_root").qvel[0:3]
        vel_ang_drone = self.data.joint("drone_root").qvel[3:6]
        vel_car = np.array([self.vitesse_car_x, self.vitesse_car_y, 0.0])
        vel_relative = vel_lin_drone - vel_car
        vitesse_relative_norme = np.linalg.norm(vel_relative)
        
        # Alignement vertical
        quat = self.data.joint("drone_root").qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Permet d'obtenir l'angle formé par le cumul du roll et pitch, servira plus tard à appliquer une pénalité si le drone penche trop.
        z_alignement = 1.0 - 2.0 * (x**2 + y**2)
        
        # --- REWARD SHAPING ANTI-ACROBATIE ---
        progres = self.ancienne_distance - vraie_distance
        self.ancienne_distance = vraie_distance
        
        reward = 0.0
        reward += progres * 10.0
        
        # 1. Tolérance Zéro sur le Tilt (Exige un vol à plat) / application dun carré pour que la pénalité deviennen exponentielle
        reward -= ((1.0 - z_alignement) ** 2) * 15.0
        
        # 2. Interdiction des rotations rapides (Anti-Flip) / de même ici sur les vitesses angulaires
        reward -= 0.1 * np.sum(np.square(vel_ang_drone))
        
        # 3. Lissage des moteurs // application d'une pénalité sur une variation trop importante des moteurs qui entrainerait des discontinuités
        delta_action = action - self.last_action
        reward -= 0.2 * np.sum(np.square(delta_action))
        self.last_action = np.copy(action)
        
        # 4. Pénalité de descente rapide
        if vel_lin_drone[2] < -1.0:
            reward -= abs(vel_lin_drone[2]) * 2.0
            
        # --- CONDITIONS TERMINALES STRICTES ---
        terminated = False
        truncated = False
        
        # Crash ou TILT INTERDIT (z_alignement < 0.8 correspond à ~36° max)
        if pos_drone[2] < 0.1 or z_alignement < 0.8:
            reward -= 200.0
            terminated = True
        # Atterrissage réussi sur cible mobile
        elif dist_xy < 0.25 and dist_z < 0.08 and vitesse_relative_norme < 0.35 and z_alignement > 0.85:
            reward += 1000.0
            terminated = True
            
        if self.current_step >= 1000:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()