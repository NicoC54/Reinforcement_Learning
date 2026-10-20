import os
import sys
import time
from stable_baselines3 import PPO

# --- 1. GESTION DES CHEMINS ---
dossier_script = os.path.dirname(os.path.abspath(__file__))
dossier_src = os.path.dirname(dossier_script)
sys.path.append(dossier_src)

from env.drone_env_static_platform import DroneLandingEnv

chemin_model = os.path.join(dossier_src, "models", "best_model", "best_model_static_platform.zip")

if not os.path.exists(chemin_model):
    print(f"❌ Fichier introuvable : {chemin_model}")
    sys.exit(1)

# --- 2. INITIALISATION (CORRIGÉE) ---
print("Initialisation du rendu MuJoCo...")
# On retire render_mode="human" pour éviter l'erreur TypeError
env = DroneLandingEnv()

# --- 3. CHARGEMENT DU MODÈLE ---
print(f"Chargement de : {chemin_model}")
model = PPO.load(chemin_model, env=env)

# --- 4. TEST DU DRONE ---
NUM_EPISODES = 10

for episode in range(1, NUM_EPISODES + 1):
    # Gym v0.26+ renvoie (obs, info)
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    
    done = False
    score = 0
    steps = 0
    
    print(f"\n--- Épisode {episode}/{NUM_EPISODES} ---")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Gestion de la compatibilité selon ta version de Gym/Gymnasium
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            
        env.render()  # L'affichage visuel 3D est appelé ici à chaque step
        
        score += reward
        steps += 1
        
        time.sleep(1 / 30.0)

    print(f"Résultat : {steps} steps | Score total : {score:.2f}")

env.close()
print("\n✅ Test terminé avec succès.")