import os
import time
import glob
from drone_env import DroneLandingEnv
from stable_baselines3 import PPO

print("Chargement de l'environnement...")
env = DroneLandingEnv()

# --- 1. CHARGEMENT AUTOMATIQUE DU MODÈLE ---
dossier_script = os.path.dirname(os.path.abspath(__file__))
fichiers_zip = glob.glob(os.path.join(dossier_script, "*.zip"))

if not fichiers_zip:
    raise FileNotFoundError("Aucun fichier modèle (.zip) trouvé dans le dossier !")

chemin_modele_recent = max(fichiers_zip, key=os.path.getctime)
print(f"Chargement automatique du cerveau le plus récent : {chemin_modele_recent}")

model = PPO.load(chemin_modele_recent)

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