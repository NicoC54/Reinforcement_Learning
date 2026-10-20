import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
# 1. Ajout des imports pour EvalCallback et CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList  # 👈 NOUVEAU

# --- 1. GESTION DES CHEMINS ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src' (ou racine du projet)
sys.path.append(dossier_src)                                  # Permet de trouver 'env'

from env.drone_env_static_platform import DroneLandingEnv

# Création automatique des dossiers de sortie
dossier_modeles = os.path.join(dossier_src, "models")
dossier_best_model = os.path.join(dossier_modeles, "best_model")  # 👈 NOUVEAU (Dossier pour le meilleur modèle)
dossier_checkpoints = os.path.join(dossier_src, "checkpoints_step1")
dossier_logs = os.path.join(dossier_src, "tb_logs")

os.makedirs(dossier_modeles, exist_ok=True)
os.makedirs(dossier_best_model, exist_ok=True)                   # 👈 NOUVEAU
os.makedirs(dossier_checkpoints, exist_ok=True)
os.makedirs(dossier_logs, exist_ok=True)

# --- 2. INSTANCIATION & VÉRIFICATION ---
print("Initialisation de l'environnement (Plateforme Statique)...")
env = DroneLandingEnv()

print("Vérification de la conformité Gymnasium...")
check_env(env)
print("Environnement validé par Stable-Baselines3 ! ✅\n")

# --- 3. CONFIGURATION DES CALLBACKS & PPO ---
# Callback 1 : Checkpoint toutes les 100 000 étapes
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=dossier_checkpoints,
    name_prefix="checkpoint_static"
)

# Callback 2 : Évaluation régulière et sauvegarde automatique du MEILLEUR modèle  👈 NOUVEAU
eval_env = DroneLandingEnv()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=dossier_best_model,
    log_path=os.path.join(dossier_logs, "eval"),
    eval_freq=10000,         # Teste le drone tous les 10 000 steps
    deterministic=True,      # Mode déterministe (sans bruit d'exploration)
    render=False
)

# Regroupement des deux callbacks dans une liste  👈 NOUVEAU
callbacks = CallbackList([checkpoint_callback, eval_callback])

print("Initialisation de l'agent PPO...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=2e-4,
    ent_coef=0.02,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log=dossier_logs
)

# --- 4. ENTRAÎNEMENT ---
TOTAL_TIMESTEPS = 1500000
print(f"Début de l'apprentissage ({TOTAL_TIMESTEPS} steps)... Appuyez sur Ctrl+C pour interrompre proprement.\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,  # 👈 NOUVEAU (Passage du groupe de callbacks)
        tb_log_name="PPO_Static_Platform"
    )
except KeyboardInterrupt:
    print("\n⚠️ Entraînement interrompu manuellement par l'utilisateur.")

finally:
    # --- 5. SAUVEGARDE FINALE ---
    chemin_sauvegarde = os.path.join(dossier_modeles, "Precise_landing_Static_platform")
    model.save(chemin_sauvegarde)
    
    # SB3 nomme TOUJOURS ce fichier 'best_model.zip'
    chemin_vrai_best_model = os.path.join(dossier_best_model, "best_model.zip")
    
    print(f"\n✅ Sauvegarde terminée !")
    print(f"📍 Modèle à la dernière étape : {chemin_sauvegarde}.zip")
    print(f"🏆 Meilleur modèle historique : {chemin_vrai_best_model}")
    print(f"📊 Suivi TensorBoard : tensorboard --logdir={dossier_logs}")