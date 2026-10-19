import os
import sys
from stable_baselines3 import PPO

# --- CORRECTION DES CHEMINS (Structure du repo) ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src'
sys.path.append(dossier_src)                                  # Permet à Python de trouver 'env'

from env.drone_env import DroneLandingEnv                     # Import correct

print("Initialisation de l'environnement...")
env = DroneLandingEnv()

# --- CHARGEMENT DU MODÈLE DEPUIS LE DOSSIER 'models' ---
dossier_modeles = os.path.join(dossier_src, "models")         # On pointe vers src/models/
chemin_modele = os.path.join(dossier_modeles, "ppo_drone_v3_smooth") 
print(f"Chargement du modèle expert lissé : {chemin_modele}.zip")

# Chargement avec liaison du nouvel environnement
model = PPO.load(chemin_modele, env=env)

print("Reprise de l'entraînement pour 500k steps supplémentaires...")
try:
    model.learn(total_timesteps=500000, reset_num_timesteps=False)
except KeyboardInterrupt:
    print("\nArrêt manuel.")
finally:
    # Sauvegarde de la mise à jour dans le dossier 'models'
    chemin_sauvegarde = os.path.join(dossier_modeles, "ppo_drone_v3_smooth")
    model.save(chemin_sauvegarde)
    print(f"Modèle mis à jour et sauvegardé sous : {chemin_sauvegarde}.zip")