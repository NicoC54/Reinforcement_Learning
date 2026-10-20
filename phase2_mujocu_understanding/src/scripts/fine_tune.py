import os
import sys
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# --- CORRECTION DES CHEMINS (Structure du repo) ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src'
sys.path.append(dossier_src)                                  # Permet à Python de trouver 'env'

from env.drone_env import DroneLandingEnv                     # Import correct

# 1. Instanciation du nouvel environnement
print("Initialisation de l'environnement avec Randomisation...")
env = DroneLandingEnv()

# 2. Recherche du modèle à charger dans le dossier 'models/'
dossier_modeles = os.path.join(dossier_src, "models")
fichiers_zip = glob.glob(os.path.join(dossier_modeles, "*.zip"))

if not fichiers_zip:
    raise FileNotFoundError(f"Aucun modèle (.zip) trouvé dans le dossier : {dossier_modeles} !")

chemin_modele_base = max(fichiers_zip, key=os.path.getctime)
print(f"Chargement du modèle de base : {chemin_modele_base}")

# 3. Chargement du modèle avec injection du nouvel environnement
model = PPO.load(chemin_modele_base, env=env)

# 4. Configuration des sauvegardes automatiques dans le dossier checkpoints_v2 de src/
dossier_checkpoints = os.path.join(dossier_src, "checkpoints_v2")
checkpoint_callback = CheckpointCallback(
    save_freq=50000, 
    save_path=dossier_checkpoints, 
    name_prefix="drone_ppo_v2"
)

# 5. Relance de l'entraînement avec protection Ctrl+C
STEPS_FINETUNING = 1000000  # Ex: 1 million de steps supplémentaires

print(f"Début du Fine-Tuning pour {STEPS_FINETUNING} steps. Appuyez sur Ctrl+C pour arrêter.")

try:
    model.learn(
        total_timesteps=STEPS_FINETUNING,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )
except KeyboardInterrupt:
    print("\nFine-tuning interrompu manuellement !")
finally:
    # Sauvegarde finale dans le dossier 'models/'
    chemin_sauvegarde_finale = os.path.join(dossier_modeles, "ppo_drone_v2_robust")
    model.save(chemin_sauvegarde_finale)
    print(f"Modèle v2 sauvegardé sous : {chemin_sauvegarde_finale}.zip")