import os
import sys
import time
import glob
from stable_baselines3 import PPO

# --- 0. CORRECTION DES CHEMINS (Structure du repo) ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src'
sys.path.append(dossier_src)                                  # Permet à Python de trouver 'env'

from env.drone_env import DroneLandingEnv                     # Import correct

print("Chargement de l'environnement...")
env = DroneLandingEnv()

# --- 1. CHARGEMENT AUTOMATIQUE DU MODÈLE DEPUIS LE DOSSIER 'models' ---
dossier_modeles = os.path.join(dossier_src, "models")         # On pointe vers src/models/
fichiers_zip = glob.glob(os.path.join(dossier_modeles, "*.zip"))

if not fichiers_zip:
    raise FileNotFoundError(f"Aucun fichier modèle (.zip) trouvé dans le dossier : {dossier_modeles} !")

chemin_modele_recent = max(fichiers_zip, key=os.path.getctime)
print(f"Chargement automatique du cerveau le plus récent : {chemin_modele_recent}")

model = PPO.load(chemin_modele_recent, env=env)

# --- 2. BOUCLE DE SIMULATION VISUELLE ---
obs, info = env.reset()
score_episode = 0.0

print("Début du vol en direct ! Appuyez sur Ctrl+C pour arrêter.")

try:
    while True:
        start_time = time.time()
        
        # L'IA choisit l'action optimale
        action, _states = model.predict(obs, deterministic=True)
        
        # Exécution dans l'environnement
        obs, reward, terminated, truncated, info = env.step(action)
        score_episode += reward
        
        # Affichage 3D
        env.render()
        
        # Synchronisation temps réel
        elapsed = time.time() - start_time
        target_time = env.model.opt.timestep * 5  # Frame skip de 5
        
        if elapsed < target_time:
            time.sleep(target_time - elapsed)
        
        # Si le drone atterrit ou se crashe, on affiche le score et on recommence
        if terminated or truncated:
            if reward > 200:
                print(f"Atterrissage RÉUSSI ! Score de l'épisode : {score_episode:.0f}")
            else:
                print(f"Crash / Échec... Score de l'épisode : {score_episode:.0f}")
                
            obs, info = env.reset()
            score_episode = 0.0  # Remise à zéro du score pour le prochain vol
            
except KeyboardInterrupt:
    print("\nSimulation arrêtée par l'utilisateur.")
finally:
    env.close()