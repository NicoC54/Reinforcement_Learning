import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class DroneLandingEnv(gym.Env):
    """
    Étape 1 : Plateforme immobile à l'origine (0, 0).
    Le drone apparaît dans un volume restreint autour de la plateforme.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Chargement de la physique
        dossier_env = os.path.dirname(os.path.abspath(__file__))
        dossier_src = os.path.dirname(dossier_env)
        chemin_xml = os.path.join(dossier_src, "xmls", "final.xml")
        
        self.model = mujoco.MjModel.from_xml_path(chemin_xml)
        self.data = mujoco.MjData(self.model)
        
        # 2. Espaces d'Action et d'Observation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Variables internes
        self.ancienne_distance = 0.0
        self.current_step = 0
        self.last_action = np.zeros(4, dtype=np.float32)
        
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
        
        # Plateforme immobile (vitesse nulle)
        vel_car = np.array([0.0, 0.0, 0.0])
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
        
        # 1. Plateforme immobile à l'origine (0,0)
        self.data.joint("Tx_car").qpos[0] = 0.0
        self.data.joint("Ty_car").qpos[0] = 0.0
        self.data.joint("Tx_car").qvel[0] = 0.0
        self.data.joint("Ty_car").qvel[0] = 0.0
        
        # 2. Génération dans une sphère/cylindre restreint autour de la cible
        rayon = np.random.uniform(0.5, 2.5)         # Distance horizontale entre 50 cm et 2.5 m
        angle = np.random.uniform(0, 2 * np.pi)     # Angle aléatoire sur 360°
        altitude = np.random.uniform(1.5, 3.5)      # Hauteur entre 1.5 m et 3.5 m
        
        x_drone = rayon * np.cos(angle)
        y_drone = rayon * np.sin(angle)
        
        # Positionnement et petites perturbations initiales (vitesse / inclinaison)
        self.data.joint("drone_root").qpos[0:3] = [x_drone, y_drone, altitude]
        self.data.joint("drone_root").qpos[3:7] = [1.0, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.0]
        self.data.joint("drone_root").qvel[0:3] = np.random.uniform(-0.2, 0.2, 3)
        
        mujoco.mj_forward(self.model, self.data)
        
        # Calcul de la distance initiale
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        self.ancienne_distance = float(np.linalg.norm(pos_drone - pos_car))
        self.last_action = np.zeros(4, dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Commandes moteurs [-1, 1] -> [0, 1]
        moteur_cmd = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        self.data.actuator("thrust_fl").ctrl[0] = moteur_cmd[0]
        self.data.actuator("thrust_fr").ctrl[0] = moteur_cmd[1]
        self.data.actuator("thrust_bl").ctrl[0] = moteur_cmd[2]
        self.data.actuator("thrust_br").ctrl[0] = moteur_cmd[3]
        
        # Simulation immobile (5 sous-pas physiques)
        for _ in range(5):
            self.data.joint("Tx_car").qvel[0] = 0.0
            self.data.joint("Ty_car").qvel[0] = 0.0
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # Extraction des positions et distances
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        
        dist_xy = np.linalg.norm(pos_drone[:2] - pos_car[:2])
        dist_z = abs(pos_drone[2] - pos_car[2])
        vraie_distance = np.linalg.norm(pos_drone - pos_car)
        
        # Vitesses
        vel_lin_drone = self.data.joint("drone_root").qvel[0:3]
        vel_ang_drone = self.data.joint("drone_root").qvel[3:6]
        vitesse_relative_norme = np.linalg.norm(vel_lin_drone)  # Vitesse voiture = 0
        
        # Alignement vertical (cosinus du tilt)
        quat = self.data.joint("drone_root").qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        z_alignement = 1.0 - 2.0 * (x**2 + y**2)
        
        # --- REWARD SHAPING ---
        progres = self.ancienne_distance - vraie_distance
        self.ancienne_distance = vraie_distance
        
        reward = 0.0
        reward += progres * 10.0
        
        # 1. Pénalité de Tilt
        reward -= ((1.0 - z_alignement) ** 2) * 15.0
        
        # 2. Anti-Flip (vitesse angulaire)
        reward -= 0.1 * np.sum(np.square(vel_ang_drone))
        
        # 3. Lissage des moteurs
        delta_action = action - self.last_action
        reward -= 0.2 * np.sum(np.square(delta_action))
        self.last_action = np.copy(action)
        
        # 4. Pénalité de descente rapide (version continue lissée)
        v_z = vel_lin_drone[2]
        if v_z < -1.0:
            reward -= (-v_z - 1.0) * 2.0
            
        # --- CONDITIONS TERMINALES ---
        terminated = False
        truncated = False
        
        # Crash ou tilt excessif (> ~36°)
        if pos_drone[2] < 0.1 or z_alignement < 0.8:
            reward -= 200.0
            terminated = True
        # Atterrissage réussi
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