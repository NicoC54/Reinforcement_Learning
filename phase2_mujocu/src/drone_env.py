import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class DroneLandingEnv(gym.Env):
    """
    Environnement Gymnasium pour l'atterrissage d'un drone sur une cible (ex. plateforme/camion).
    """
    def __init__(self):
        super().__init__()
        
        # 1. Chargement de la physique (MuJoCo)
        dossier_script = os.path.dirname(os.path.abspath(__file__))
        chemin_xml = os.path.join(dossier_script, "xmls", "final.xml")
        
        self.model = mujoco.MjModel.from_xml_path(chemin_xml)
        self.data = mujoco.MjData(self.model)
        
        # Affichage de la masse détectée
        drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Drone")
        masse_drone = self.model.body_subtreemass[drone_body_id]
        print(f"-> Masse totale du drone détectée par MuJoCo : {masse_drone:.3f} kg")
        
        # 2. Espace d'Action : 4 moteurs commandés de -1.0 à 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 3. Espace d'Observation : 13 variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Paramètres de rendu
        self.metadata = {"render_modes": ["human"]}
        self.viewer = None

    def _get_obs(self):
        pos_drone = self.data.body("Drone").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        
        pos_relative = pos_drone - pos_car
        
        vel_lin_drone = self.data.joint("drone_root").qvel[0:3] 
        vel_ang_drone = self.data.joint("drone_root").qvel[3:6]
        vel_relative = vel_lin_drone - np.array([0.0, 0.0, 0.0])

        quat_drone = self.data.body("Drone").xquat

        obs = np.concatenate([
            pos_relative / 10.0,
            vel_relative / 5.0,
            quat_drone,
            vel_ang_drone / 10.0
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. Position initiale aléatoire (ex: entre -2m et +2m en XY, et 1.5m à 3.5m en Z)
        pos_x = np.random.uniform(-2.0, 2.0)
        pos_y = np.random.uniform(-2.0, 2.0)
        pos_z = np.random.uniform(1.5, 3.5)
        self.data.joint("drone_root").qpos[0:3] = [pos_x, pos_y, pos_z]
        
        # 2. Inclinaison initiale aléatoire (légère désorientation)
        tilt_x = np.random.uniform(-0.15, 0.15)
        tilt_y = np.random.uniform(-0.15, 0.15)
        self.data.joint("drone_root").qpos[3:7] = [1.0, tilt_x, tilt_y, 0.0]
        
        # 3. Vitesse initiale aléatoire (impulsion au départ)
        self.data.joint("drone_root").qvel[0:3] = np.random.uniform(-0.3, 0.3, 3)
        
        mujoco.mj_forward(self.model, self.data)
        
        # Mise à jour de la distance initiale pour le calcul du progrès
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        self.ancienne_distance = float(np.linalg.norm(pos_drone - pos_car))
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Action : Mapping [-1, 1] vers [0, 1]
        moteur_cmd = np.clip((action + 1.0) / 2.0, 0.0, 1.0)
        
        self.data.actuator("thrust_fl").ctrl[0] = moteur_cmd[0]
        self.data.actuator("thrust_fr").ctrl[0] = moteur_cmd[1]
        self.data.actuator("thrust_bl").ctrl[0] = moteur_cmd[2]
        self.data.actuator("thrust_br").ctrl[0] = moteur_cmd[3]
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # Extraction des données
        pos_drone = self.data.body("drone_frame").xpos
        pos_car = self.data.geom("Landing_box_h").xpos
        vel_lin = self.data.joint("drone_root").qvel[0:3]
        
        # Distances
        dist_xy = np.linalg.norm(pos_drone[:2] - pos_car[:2])
        dist_z = abs(pos_drone[2] - pos_car[2])
        vraie_distance = np.linalg.norm(pos_drone - pos_car)
        vraie_vitesse = np.linalg.norm(vel_lin)
        
        # Alignement vertical
        quat = self.data.joint("drone_root").qpos[3:7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        z_alignement = 1.0 - 2.0 * (x**2 + y**2) # 1.0 = à plat
        
        # ----------------------------------------------------
        # REWARD SHAPING : Approche Multi-Objectifs
        # ----------------------------------------------------
        reward = 0.0
        
        # 1. Progrès vers la cible (Dense Reward)
        # Très incitatif pour guider l'agent au début
        progres = self.ancienne_distance - vraie_distance
        self.ancienne_distance = vraie_distance
        reward += progres * 10.0 
        
        # 2. Maintien de l'horizontalité (Pénalité)
        # Exponentiel : petite erreur = petite pénalité, grosse erreur = grosse pénalité
        erreur_tilt = 1.0 - z_alignement
        reward -= (erreur_tilt ** 2) * 5.0
        
        # 3. Contrôle de la vitesse de descente (Pénalité)
        # On pénalise si le drone tombe trop vite (Z velocity < -1.0 m/s)
        if vel_lin[2] < -1.0:
            reward -= abs(vel_lin[2]) * 2.0
            
        # 4. Économie d'énergie / Lissage des commandes
        # Décourage l'oscillation des moteurs
        reward -= 0.05 * np.sum(np.square(action))
        
        # ----------------------------------------------------
        # CONDITIONS TERMINALES (Sparse Rewards)
        # ----------------------------------------------------
        terminated = False
        truncated = False
        
        # Crash / Renversement
        if pos_drone[2] < 0.1 or z_alignement < 0.3:
            reward -= 200.0
            terminated = True
            
        # Atterrissage Réussi
        # Proche au sol, centré, lent, et droit
        elif dist_xy < 0.15 and dist_z < 0.05 and vraie_vitesse < 0.2 and z_alignement > 0.9:
            reward += 500.0
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


if __name__ == "__main__":
    env = DroneLandingEnv()
    obs, info = env.reset()
    print(f"Forme de l'observation : {obs.shape}")
    print(f"Valeurs initiales : \n{obs}")