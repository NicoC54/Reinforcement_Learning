import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# --- CORRECTION DES CHEMINS (Structure du repo) ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src'
sys.path.append(dossier_src)                                  # Permet à Python de trouver 'env'

from env.drone_env import DroneLandingEnv                     # Import correct

# 1. Chargement de l'environnement
env = DroneLandingEnv()

# 2. Chargement du modèle depuis le dossier 'models/'
dossier_modeles = os.path.join(dossier_src, "models")
nom_modele = "ppo_drone_v3_smooth" 
chemin_modele = os.path.join(dossier_modeles, f"{nom_modele}.zip")

print(f"Chargement du modèle : {chemin_modele}")
model = PPO.load(chemin_modele)

# Paramètres du test
N_EPISODES = 20
succes = 0
scores = []
temps_vol = []

print(f"\n--- Début du test sur {N_EPISODES} atterrissages mobiles sinusoïdaux ---\n")

try:
    for ep in range(1, N_EPISODES + 1):
        obs, info = env.reset()
        done = False
        score_episode = 0.0
        steps = 0
        
        while not done:
            start_time = time.time()
            
            # Action pure (sans le bruit aléatoire de l'exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            score_episode += reward
            steps += 1
            done = terminated or truncated
            
            # Rendu 3D
            env.render()
            
            # Maintien de la vitesse temps réel (1:1)
            elapsed = time.time() - start_time
            target_time = env.model.opt.timestep * 5  # 0.01s par step
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        
        scores.append(score_episode)
        duree_sec = steps * 0.01
        temps_vol.append(duree_sec)
        
        # Validation du succès selon le score
        if score_episode > 500:
            succes += 1
            print(f"Épisode {ep:02d}/{N_EPISODES} : RÉUSSI   | Score: {score_episode:4.0f} | Temps: {duree_sec:.2f}s")
        else:
            print(f"Épisode {ep:02d}/{N_EPISODES} : ÉCHEC     | Score: {score_episode:4.0f} | Temps: {duree_sec:.2f}s")

except KeyboardInterrupt:
    print("\nTest interrompu manuellement.")
finally:
    env.close()

# --- STATISTIQUES FINALES ---
if scores:
    print("\n" + "=" * 45)
    print(f" BILAN DES TESTS ({len(scores)} épisodes)")
    print("=" * 45)
    print(f" Taux de réussite : {(succes / len(scores)) * 100:.1f} %")
    print(f" Score moyen      : {np.mean(scores):.1f}")
    print(f" Temps moyen/vol  : {np.mean(temps_vol):.2f} s")
    print("=" * 45)