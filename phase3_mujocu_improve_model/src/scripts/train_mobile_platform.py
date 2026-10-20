import os
import sys
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Ajout du dossier src au PYTHONPATH pour importer l'environnement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation de l'environnement mobile
from env.drone_env_mobile_platform import DroneLandingEnv

class SaveBestModelCustomNameCallback(EvalCallback):
    """
    Callback personnalisé pour renommer automatiquement le meilleur modèle 
    en 'best_model_mobile_platform.zip' lors de son enregistrement.
    """
    def __init__(self, *args, custom_model_name="best_model_mobile_platform.zip", **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_model_name = custom_model_name

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        # Si un nouveau meilleur modèle vient d'être sauvegardé par la classe parente
        if self.best_model_save_path is not None:
            default_best_path = os.path.join(self.best_model_save_path, "best_model.zip")
            custom_best_path = os.path.join(self.best_model_save_path, self.custom_model_name)
            
            if os.path.exists(default_best_path):
                # On duplique/renomme le fichier sous le nom personnalisé
                shutil.copy(default_best_path, custom_best_path)
                
        return continue_training

if __name__ == "__main__":
    # --- 1. Chemins et dossiers ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ancien_modele_path = os.path.join(base_dir, "models_mobile", "best_model", "best_model_mobile_platform.zip")    
    log_dir = os.path.join(base_dir, "tb_logs_mobile")
    save_dir = os.path.join(base_dir, "models_mobile")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. Initialisation des environnements ---
    env = Monitor(DroneLandingEnv())
    eval_env = Monitor(DroneLandingEnv())

    # --- 3. Configuration des Callbacks ---
    # On utilise notre callback personnalisé à la place de l'EvalCallback standard
    eval_callback = SaveBestModelCustomNameCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
        custom_model_name="best_model_mobile_platform.zip"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_mobile_ckpt"
    )

    # --- 4. Chargement du modèle pour le Fine-Tuning ---
    if not os.path.exists(ancien_modele_path):
        print(f"❌ Ancien modèle introuvable : {ancien_modele_path}")
        sys.exit(1)
        
    print(f"✅ Chargement des poids depuis : {ancien_modele_path}")
    
    model = PPO.load(
        ancien_modele_path, 
        env=env, 
        tensorboard_log=log_dir,
        custom_objects={"learning_rate": 0.0001} 
    )

    # --- 5. Lancement de l'apprentissage ---
    print("🚀 Début du fine-tuning sur la plateforme mobile...")
    
    model.learn(
        total_timesteps=1_500_000, 
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="ppo_mobile_run",
        reset_num_timesteps=False
    )

    # --- 6. Sauvegarde finale ---
    final_path = os.path.join(save_dir, "final_model_mobile")
    model.save(final_path)
    print(f"🏁 Entraînement terminé ! Modèle final sauvegardé sous : {final_path}")