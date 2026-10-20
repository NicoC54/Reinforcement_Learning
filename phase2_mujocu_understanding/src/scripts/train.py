import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# --- CORRECTION DES CHEMINS (Structure du repo) ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src'
sys.path.append(dossier_src)                                  # Permet à Python de trouver 'env'

from env.drone_env import DroneLandingEnv                     # Import correct

# 1. Instanciation de l'environnement
env = DroneLandingEnv()

# 2. Vérification automatique de conformité Gymnasium
print("Vérification de l'environnement...")
check_env(env)
print("Environnement validé par Stable-Baselines3 ! ✅\n")

# 3. Initialisation de l'algorithme PPO
print("Initialisation du réseau de neurones PPO...")
model = PPO("MlpPolicy", env, verbose=1)

# 4. Lancement de l'entraînement
print("Début du grand entraînement (500 000 d'étapes)...")
model.learn(total_timesteps=500000)

# 5. Sauvegarde du modèle entraîné dans le dossier 'models/'
dossier_modeles = os.path.join(dossier_src, "models")
chemin_sauvegarde = os.path.join(dossier_modeles, "ppo_drone_v1")
model.save(chemin_sauvegarde)

print(f"\nEntraînement terminé avec succès !")
print(f"Le modèle a été sauvegardé sous : {chemin_sauvegarde}.zip")