import os
import glob
from drone_env import DroneLandingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. Instanciation du nouvel environnement (avec le reset() plus difficile)
print("Initialisation de l'environnement avec Randomisation...")
env = DroneLandingEnv()

# 2. Recherche du modèle v1 / plus récent à charger
dossier_script = os.path.dirname(os.path.abspath(__file__))
fichiers_zip = glob.glob(os.path.join(dossier_script, "*.zip"))

if not fichiers_zip:
    raise FileNotFoundError("Aucun modèle (.zip) trouvé pour démarrer le fine-tuning !")

chemin_modele_base = max(fichiers_zip, key=os.path.getctime)
print(f"Chargement du modèle de base : {chemin_modele_base}")

# 3. Chargement du modèle avec injection du nouvel environnement
# IMPORTANT : Toujours passer env=env pour lier le nouveau reset() au modèle
model = PPO.load(chemin_modele_base, env=env)

# 4. Configuration des sauvegardes automatiques
dossier_checkpoints = os.path.join(dossier_script, "checkpoints_v2")
checkpoint_callback = CheckpointCallback(
    save_freq=50000, 
    save_path=dossier_checkpoints, 
    name_prefix="drone_ppo_v2"
)

# 5. Relance de l'entraînement avec protection Ctrl+C
STEPS_FINETUNING = 1000000  # Ex: 1 million de steps supplémentaires

print(f"Début du Fine-Tuning pour {STEPS_FINETUNING} steps. Appuyez sur Ctrl+C pour arrêter.")

try:
    # reset_num_timesteps=False permet de continuer la courbe TensorBoard/logs là où elle s'était arrêtée
    model.learn(
        total_timesteps=STEPS_FINETUNING,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )
except KeyboardInterrupt:
    print("\nFine-tuning interrompu manuellement !")
finally:
    chemin_sauvegarde_finale = os.path.join(dossier_script, "ppo_drone_v2_robust")
    model.save(chemin_sauvegarde_finale)
    print(f"Modèle v2 sauvegardé sous : {chemin_sauvegarde_finale}.zip")