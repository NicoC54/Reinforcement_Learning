import numpy as np

class DronePhysics:
    def __init__(self):
        # 1. Constantes physiques
        self.m = 1.0
        self.g = 9.81
        self.L = 0.2
        self.d = self.L / np.sqrt(2.0)  # Calcul géométrique pour la config en X
        self.c_tau = 0.01               # Équivalent de votre k_yaw
        
        self.I = np.diag([0.01, 0.01, 0.02])
        self.I_inv = np.linalg.inv(self.I)
        self.dt = 0.01
        
        # 2. Vecteur d'état initial (13 valeurs à 0.0)
        self.state = np.zeros(13, dtype=np.float64)
        self.state[6] = 1.0  # qw = 1 (quaternion identité, drone à plat)

        # 3. Matrice de mixage (Config X, Repère FRD) (respectivement ordre des moteurs : 1 : bas_droit,2 : haut_droit, 3 : bas_gauche, 4 : haut_gauche )
        self.M_mix = np.array([
            [-1.0,        -1.0,        -1.0,        -1.0       ], # Force Z
            [-self.d,     -self.d,      self.d,      self.d    ], # Couple Roulis (Roll)
            [-self.d,      self.d,     -self.d,      self.d    ], # Couple Tangage (Pitch)
            [-self.c_tau,  self.c_tau,  self.c_tau, -self.c_tau]  # Couple Lacet (Yaw)
        ], dtype=np.float64)

    # --- VUES LISIBLES (Propriétés) ---
    @property
    def position(self):
        return self.state[0:3]

    @property
    def vitesse(self):
        return self.state[3:6]

    @property
    def quaternion(self):
        return self.state[6:10]

    @property
    def vitesses_angulaires(self):
        return self.state[10:13]

    # --- UTILITAIRES ---
    @staticmethod
    def quat_to_rot_matrix(q):
        """Convertit un quaternion unitaire [qw, qx, qy, qz] en matrice 3x3."""
        qw, qx, qy, qz = q
        qx2, qy2, qz2 = qx*qx, qy*qy, qz*qz
        qwx, qwy, qwz = qw*qx, qw*qy, qw*qz
        qxy, qxz, qyz = qx*qy, qx*qz, qy*qz

        return np.array([
            [1.0 - 2.0*(qy2 + qz2), 2.0*(qxy - qwz),       2.0*(qxz + qwy)],
            [2.0*(qxy + qwz),       1.0 - 2.0*(qx2 + qz2), 2.0*(qyz - qwx)],
            [2.0*(qxz - qwy),       2.0*(qyz + qwx),       1.0 - 2.0*(qx2 + qy2)]
        ], dtype=np.float64)

    def getForceAndMoments(self, thrusts):
        """Prend [f1, f2, f3, f4], retourne [Fz, tau_x, tau_y, tau_z]"""
        return self.M_mix @ thrusts

    # --- LE COEUR MATHÉMATIQUE (PFD et TDM) ---
    def derivatives(self, state, thrusts):
        """Calcule les dérivées temporelles de l'état."""
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        # 1. Obtenir les forces et couples
        f_and_m = self.getForceAndMoments(thrusts)
        Fz = f_and_m[0]
        tau = f_and_m[1:4]

        # 2. PFD (Translation) dans le repère Monde (NED)
        R = self.quat_to_rot_matrix(q)
        thrust_body = np.array([0.0, 0.0, Fz])
        thrust_world = R @ thrust_body
        v_dot = (1.0 / self.m) * thrust_world + np.array([0.0, 0.0, self.g])

        # 3. TDM (Rotation) dans le repère Corps (FRD) avec formule de Bour
        L_body = self.I @ omega
        gyroscopic = np.cross(omega, L_body)
        omega_dot = self.I_inv @ (tau - gyroscopic)

        # 4. Cinématique du quaternion
        qw, qx, qy, qz = q
        p, q_rate, r = omega
        q_dot = 0.5 * np.array([
            -qx*p - qy*q_rate - qz*r,
             qw*p + qy*r      - qz*q_rate,
             qw*q_rate - qx*r + qz*p,
             qw*r + qx*q_rate - qy*p
        ])

        # Retourne les 13 dérivées concaténées : [p_dot, v_dot, q_dot, omega_dot]
        return np.concatenate([v, v_dot, q_dot, omega_dot])

    # --- L'INTÉGRATEUR NUMÉRIQUE ---
    def step_rk4(self, thrusts):
        """Avance la simulation d'un pas de temps dt avec Runge-Kutta 4."""
        k1 = self.derivatives(self.state, thrusts)
        k2 = self.derivatives(self.state + 0.5 * self.dt * k1, thrusts)
        k3 = self.derivatives(self.state + 0.5 * self.dt * k2, thrusts)
        k4 = self.derivatives(self.state + self.dt * k3, thrusts)

        # Mise à jour de l'état
        self.state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Renormalisation du quaternion (Crucial !)
        self.state[6:10] /= np.linalg.norm(self.state[6:10])



if __name__ == "__main__":
    drone = DronePhysics()
    
    # 1. Calcul de la poussée de stationnaire (Hover)
    # Dans le repère FRD, la poussée totale Fz doit compenser mg.
    # Chaque moteur doit fournir (m * g) / 4
    hover_thrust = (drone.m * drone.g) / 4.0
    thrusts = np.array([hover_thrust+ 0.5, hover_thrust+ 0.5, hover_thrust+ 0.5, hover_thrust+ 0.5])

    print("--- Test de Vol Stationnaire ---")
    print(f"Altitude initiale : {drone.position[2]:.4f} m")
    
    # On fait tourner la simulation pendant 1 seconde (100 pas de dt=0.01)
    for _ in range(100):
        drone.step_rk4(thrusts)
        
    print(f"Altitude après 1s : {drone.position[2]:.4f} m")
    print(f"Vitesse en Z      : {drone.vitesse[2]:.4f} m/s")
    # Résultat attendu : Altitude 0.0000 m, Vitesse 0.0000 m/s (Immobile !)