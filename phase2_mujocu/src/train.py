import os
from drone_env import DroneLandingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
print("Début du grand entraînement (1 000 000 d'étapes)...")
model.learn(total_timesteps=500000)

# 5. Sauvegarde du modèle entraîné
dossier_script = os.path.dirname(os.path.abspath(__file__))
chemin_sauvegarde = os.path.join(dossier_script, "ppo_drone_v1")
model.save(chemin_sauvegarde)

print(f"\nEntraînement terminé avec succès !")
print(f"Le modèle a été sauvegardé sous : {chemin_sauvegarde}.zip")