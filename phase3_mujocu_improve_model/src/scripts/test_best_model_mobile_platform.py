import os
import sys
import time
from stable_baselines3 import PPO

# --- 1. GESTION DES CHEMINS ---
dossier_script = os.path.dirname(os.path.abspath(__file__))
dossier_src = os.path.dirname(dossier_script)
sys.path.append(dossier_src)

# Importation de l'environnement avec la plateforme mobile
from env.drone_env_mobile_platform import DroneLandingEnv

# Chemin vers le nouveau modèle mobile (attention au dossier models_mobile)
chemin_model = os.path.join(dossier_src, "models_mobile", "best_model.zip")

if not os.path.exists(chemin_model):
    print(f"❌ Fichier introuvable : {chemin_model}")
    sys.exit(1)

# --- 2. INITIALISATION ---
print("Initialisation du rendu MuJoCo (Plateforme Mobile)...")
env = DroneLandingEnv()

# --- 3. CHARGEMENT DU MODÈLE ---
print(f"Chargement de : {chemin_model}")
model = PPO.load(chemin_model, env=env)

# --- 4. TEST DU DRONE ---
NUM_EPISODES = 30

for episode in range(1, NUM_EPISODES + 1):
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    
    done = False
    score = 0
    steps = 0
    
    print(f"\n--- Épisode {episode}/{NUM_EPISODES} ---")
    
    while not done:
        # deterministic=True garantit que le modèle utilise sa politique optimale sans bruit aléatoire
        action, _states = model.predict(obs, deterministic=True)
        
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            
        env.render()  
        
        score += reward
        steps += 1
        
        # 60 FPS pour un rendu visuel fluide et réaliste
        time.sleep(1 / 60.0)

    print(f"Résultat : {steps} steps | Score total : {score:.2f}")

env.close()
print("\n✅ Test sur cible mobile terminé avec succès.")