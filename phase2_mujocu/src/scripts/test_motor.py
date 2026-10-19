import os
import mujoco
import mujoco.viewer
import time

# --- CORRECTION DU CHEMIN DU XML ---
dossier_script = os.path.dirname(os.path.abspath(__file__))   # Dossier 'scripts'
dossier_src = os.path.dirname(dossier_script)                 # Dossier 'src'
chemin_xml = os.path.join(dossier_src, "xmls", "final.xml")   # Pointe vers src/xmls/final.xml

model = mujoco.MjModel.from_xml_path(chemin_xml)
data = mujoco.MjData(model)

print("Lancement de MuJoCo...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    moteurs = ["thrust_fl", "thrust_fr", "thrust_bl", "thrust_br"]
    
    # --- PHASE 1 : TEST INDIVIDUEL DES MOTEURS ---
    for nom_moteur in moteurs:
        print(f"\n---> TEST DE : {nom_moteur} (Poussée active pendant 3 sec)")
        
        # Réinitialisation de la position du drone à chaque moteur
        mujoco.mj_resetData(model, data)
        
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 3.0:
            
            # 1. On remet toutes les commandes à 0
            data.ctrl[:] = 0.0
            
            # 2. On allume SEULEMENT le moteur en cours
            data.actuator(nom_moteur).ctrl[0] = 0.5
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    # --- PHASE 2 : TEST DES 4 MOTEURS EN MÊME TEMPS ---
    print("\n---> TEST FINAL : Les 4 moteurs simultanément !")
    print("Fermez la fenêtre 3D pour arrêter le script.")
    
    # On remet le drone au centre avant le grand décollage
    mujoco.mj_resetData(model, data)
    
    # Cette boucle tourne à l'infini, tant que vous ne fermez pas la fenêtre
    while viewer.is_running():
        
        # On allume les 4 moteurs en même temps (ici à 0.5)
        data.ctrl[:] = 0.5 
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Utiliser le timestep du modèle est plus précis que 0.01
        time.sleep(model.opt.timestep) 

print("Test terminé.")