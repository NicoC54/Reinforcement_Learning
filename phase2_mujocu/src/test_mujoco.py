import os
import mujoco
import mujoco.viewer
import time

# 1. Utilisation d'un chemin robuste (comme vu précédemment)
dossier_script = os.path.dirname(os.path.abspath(__file__))
chemin_xml = os.path.join(dossier_script, "xmls", "final.xml")

model = mujoco.MjModel.from_xml_path(chemin_xml)
data = mujoco.MjData(model)

# 2. Lancement de la fenêtre 3D interactive
print("Lancement de MuJoCo...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        # --- CORRECTION : On cible les moteurs par leur ID ---
        # (Remarque: Il faudra vérifier les noms exacts des actionneurs dans vos sous-fichiers XML)
        try:
            # Exemple : Si vos actionneurs s'appellent motor1, motor2, etc.
            data.actuator("thrust_fl").ctrl = 0.6
            data.actuator("thrust_fr").ctrl = 0.6
            data.actuator("thrust_bl").ctrl = 0.6
            data.actuator("thrust_br").ctrl = 0.6
        except KeyError:
            # Si les noms ne sont pas bons, on ne fait rien (les gaz restent à 0)
            pass 
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)