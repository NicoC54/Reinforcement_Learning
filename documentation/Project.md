Voici le **cahier des charges technique ultra-détaillé** de votre projet hybride. L'objectif est que vous sachiez exactement **quelles variables créer, quelles fonctions écrire, et comment tester** chaque brique sans jamais vous retrouver bloqué devant votre écran.

---

# PARTIE 1 : La Phase Théorique ("Fait Maison" en NumPy)

*Objectif : Valider vos compétences mathématiques (PFD, TDM, RK4) et faire voler un drone en stationnaire.*

1. **Jalon 1 : Coder le Moteur Physique 3D de zéro:** Fichier : phase1_theory/drone_physics.py.
Vous allez créer une classe `DronePhysics` qui ne dépend que de `numpy`.

**1. Les constantes physiques à définir dans `__init__` :**

* `m = 1.0` (Masse en kg)
* `g = 9.81` (Gravité en $\text{m/s}^2$)
* `L = 0.2` (Demi-envergure : distance entre le centre du drone et un moteur, en mètres)
* `k_yaw = 0.01` (Coefficient de traînée aérodynamique pour le lacet)
* `I = np.diag([0.01, 0.01, 0.02])` (Matrice d'inertie $3 \times 3$ en $\text{kg}\cdot\text{m}^2$, et précalculez son inverse `I_inv = np.linalg.inv(I)`)
* `dt = 0.01` (Pas de temps de simulation : $100\text{ Hz}$)

**2. Le vecteur d'état `state` (vecteur NumPy de taille 13) :**

* Indices `0:3` : Position dans le monde $\mathbf{p} = [x, y, z]$
* Indices `3:6` : Vitesse linéaire dans le monde $\mathbf{v} = [v_x, v_y, v_z]$
* Indices `6:10` : Quaternion d'orientation $\mathbf{q} = [q_w, q_x, q_y, q_z]$ (initialisé à `[1, 0, 0, 0]` pour "à plat")
* Indices `10:13` : Vitesse angulaire dans le repère du drone $\boldsymbol{\omega} = [p, q, r]$

**3. Les 4 méthodes à coder :**

* `quat_to_rot_matrix(q)` : Prend $[q_w, q_x, q_y, q_z]$ et retourne la matrice $R$ ($3 \times 3$).
* `motor_mixing(thrusts)` : Prend les 4 forces $[f_1, f_2, f_3, f_4]$ (en Newtons) d'un drone en configuration "+" (1: Avant, 2: Droite, 3: Arrière, 4: Gauche) et retourne :
* $F_z = f_1 + f_2 + f_3 + f_4$
* $\tau_x = L (f_4 - f_2)$ (Roulis : gauche moins droite)
* $\tau_y = L (f_3 - f_1)$ (Tangage : arrière moins avant)
* $\tau_z = k_{\text{yaw}} (f_1 - f_2 + f_3 - f_4)$ (Lacet : moteurs horaires vs anti-horaires)


* `derivatives(state, thrusts)` : C'est le cœur mathématique ! Cette fonction calcule la dérivée $\dot{X}$ des 13 valeurs :
* $\dot{\mathbf{p}} = \mathbf{v}$
* $\dot{\mathbf{v}} = \frac{1}{m} R(\mathbf{q}) \begin{pmatrix} 0 \\ 0 \\ F_z \end{pmatrix} - \begin{pmatrix} 0 \\ 0 \\ g \end{pmatrix}$ *(Votre PFD)*
* $\dot{\mathbf{q}} = \frac{1}{2} \begin{pmatrix} -q_x p - q_y q - q_z r \\ q_w p + q_y r - q_z q \\ q_w q - q_x r + q_z p \\ q_w r + q_x q - q_y p \end{pmatrix}$ *(Cinématique du quaternion)*
* $\dot{\boldsymbol{\omega}} = I^{-1} (\boldsymbol{\tau} - \boldsymbol{\omega} \times (I \boldsymbol{\omega}))$ *(Votre TDM, utilisez `np.cross`)*


* `step_rk4(thrusts)` : Applique la formule de Runge-Kutta 4 sur `derivatives`, met à jour `self.state`, et **renormalise le quaternion** (`state[6:10] /= np.linalg.norm(state[6:10])`).

**Critère de réussite :** Appelez `step_rk4([2.4525, 2.4525, 2.4525, 2.4525])` 100 fois. Comme $4 \times 2.4525 = 9.81\text{ N}$, le drone doit rester exactement immobile à son altitude de départ.


2. **Jalon 2 : Le Visualiseur 3D de Débogage:** Fichier : phase1_theory/visualizer.py.
Avant de brancher l'IA, vous devez voir votre drone. Créez une classe `DroneVisualizer` avec `matplotlib.pyplot`.

**Comment dessiner le drone en 3D :**

* Dans le repère du drone, le moteur Avant est à $[L, 0, 0]$ et l'Arrière à $[-L, 0, 0]$.
* Pour connaître leur position dans le Monde, appliquez : $\mathbf{p}_{\text{moteur\_monde}} = \mathbf{p} + R(\mathbf{q}) \cdot \mathbf{p}_{\text{moteur\_corps}}$.
* Tracez un segment rouge entre Avant et Arrière, et un segment bleu entre Gauche et Droite.

**Critère de réussite :** Faites une boucle avec une poussée légèrement plus forte sur le moteur Gauche (`f4 = 2.6` au lieu de `2.4525`). Vous devez voir la croix s'incliner vers la droite et partir sur le côté en tombant.


3. **Jalon 3 : Votre Premier RL (Vol Stationnaire):** Fichiers : hover_env.py et train_hover.py.
Vous encapsulez votre classe `DronePhysics` dans un environnement `gymnasium.Env`.

**1. Configuration de l'environnement (`hover_env.py`) :**

* `action_space` : `spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)`.
* **Conversion de l'action :** Le réseau sort $a_i \in [-1, 1]$. Convertissez-le en poussée autour du point d'équilibre : $f_i = \frac{mg}{4} + a_i \times 2.0\text{ N}$. (Astuce : centrer l'action autour du vol stationnaire accélère l'apprentissage par 10 !).
* `observation_space` : Un vecteur de 12 valeurs. Au lieu du quaternion (parfois abstrait pour le réseau), donnez : l'erreur de position $(\mathbf{p} - \mathbf{p}_{\text{cible}})$, la vitesse $\mathbf{v}$, la 3ème ligne de la matrice de rotation $R[2, :]$ (qui indique où pointe le haut du drone), et la vitesse angulaire $\boldsymbol{\omega}$.
* `reward` : `r = 2.0 - 1.5 * norm(p - p_cible) - 0.1 * norm(v) - 0.1 * norm(omega) - 0.05 * norm(action)`.
* `terminated` (Conditions de crash) : `True` si $z < 0.1\text{ m}$, ou si $\Vert{}\mathbf{p} - \mathbf{p}_{\text{cible}}\Vert{} > 4.0\text{ m}$, ou si le drone est penché à plus de 60° ($R_{2,2} < 0.5$).

**2. Entraînement (`train_hover.py`) :**

* Importez `PPO` depuis `stable_baselines3`.
* Lancez `model = PPO("MlpPolicy", env, verbose=1)` puis `model.learn(total_timesteps=300_000)`.

**Critère de réussite :** La récompense moyenne par épisode (`ep_rew_mean` dans le terminal) passe du négatif à une valeur positive stable. Votre drone tient tout seul en l'air ! **Fin de la Partie 1.**


---

# PARTIE 2 : La Phase Industrielle (MuJoCo & Atterrissage Mobile)

*Objectif : Utiliser le moteur physique standard de l'industrie pour gérer la dynamique de contact, le chariot mobile et la robustesse Sim-to-Real.*

1. **Jalon 4 : Modélisation MJCF (MuJoCo XML):** Fichier : phase2_industry/assets/scene.xml.
Installez MuJoCo (`pip install mujoco`). Vous n'écrivez plus d'équations, vous décrivez la scène dans un fichier XML.

**1. Ce que doit contenir votre fichier `scene.xml` :**

* **Le Sol :** Un `<geom type="plane" size="10 10 0.1"/>`.
* **Le Chariot (`<body name="cart" pos="0 0 0.1">`) :**
* Pour qu'il bouge uniquement sur le sol $(X, Y)$, donnez-lui deux joints glissières : `<joint name="cart_x" type="slide" axis="1 0 0"/>` et `<joint name="cart_y" type="slide" axis="0 1 0"/>`.
* Ajoutez une géométrie plate (le plateau d'atterrissage) : `<geom name="pad" type="cylinder" size="0.4 0.02" mass="5.0" friction="1.0 0.005 0.0001"/>`.


* **Le Drone (`<body name="drone" pos="0 0 2.0">`) :**
* Donnez-lui 6 degrés de liberté avec `<freejoint/>`.
* Créez le châssis central, 4 bras, et 4 petits pieds (`<geom type="sphere" size="0.02"/>`) sous les moteurs pour amortir le contact avec le chariot.
* Placez 4 `<site>` à l'emplacement des hélices (`motor1`, `motor2`, `motor3`, `motor4`).


* **Les Actionneurs (`<actuator>`) :**
* Déclarez 4 moteurs qui poussent sur les sites : `<general name="thrust1" site="motor1" gear="0 0 1 0 0 0.01" ctrlrange="0 6"/>` (le `gear` applique une force en $Z$ et un petit couple de lacet autour de $Z$).
* Déclarez 2 moteurs pour piloter le chariot : `<velocity name="cart_vx" joint="cart_x"/>` et `<velocity name="cart_vy" joint="cart_y"/>`.



**Critère de réussite :** Lancez `python -m mujoco.viewer --mjcf=scene.xml`. Une fenêtre 3D s'ouvre : vous voyez votre drone tomber et rebondir physiquement sur le plateau du chariot, avec des ombres et une gestion parfaite des contacts.


2. **Jalon 5 : L'Environnement d'Atterrissage et le Chariot Aléatoire:** Fichier : phase2_industry/mujoco_landing_env.py.
Vous créez la classe `MujocoLandingEnv(gymnasium.Env)` qui pilote la simulation MuJoCo (`mj_step`).

**1. Animer le chariot de façon réaliste (Mouvement de Markov / Ornstein-Uhlenbeck) :**

* Si la vitesse du chariot change brutalement à chaque milliseconde, c'est irréaliste.
* Dans `step()`, faites évoluer la vitesse cible du chariot doucement : `v_target += -0.5 * v_target * dt + np.random.normal(0, 0.5, size=2) * np.sqrt(dt)`. Envoyez cette vitesse aux actionneurs `cart_vx` et `cart_vy`.

**2. Le Frame Slicing (Physique à 500 Hz, RL à 50 Hz) :**

* MuJoCo a besoin de petits pas de temps pour gérer les rebonds (`dt_sim = 0.002s`).
* Mais le réseau de neurones n'a pas besoin de réfléchir 500 fois par seconde. Dans la fonction `step(action)` de Gymnasium, faites une boucle `for _ in range(10): mujoco.mj_step(model, data)` avec la même action. Le RL agit ainsi à $50\text{ Hz}$ ($0.02\text{s}$).

**3. L'Espace d'Observation (15 valeurs dans le repère relatif) :**

* $\Delta \mathbf{p} = \mathbf{p}_{\text{drone}} - \mathbf{p}_{\text{chariot}}$ (3 valeurs : distance 3D à la cible)
* $\Delta \mathbf{v} = \mathbf{v}_{\text{drone}} - \mathbf{v}_{\text{chariot}}$ (3 valeurs : vitesse relative)
* Matrice de rotation du drone (ou Roulis, Tangage, Lacet + vitesses angulaires : 6 valeurs)
* Accélération actuelle du chariot $(a_{cx}, a_{cy})$ (2 valeurs : permet au drone d'anticiper les virages)
* Booléen de contact (1 valeur : `1.0` si les patins du drone touchent le plateau du chariot, `0.0` sinon, lu via `data.contact`).

**4. La Fonction de Récompense en "Cône d'Approche" (Le secret de la réussite) :**

* Calculez la distance horizontale $d_{xy} = \sqrt{\Delta x^2 + \Delta y^2}$ et la hauteur relative $d_z = \Delta z$.
* **Idée du cône :** Tant que $d_{xy}$ est grand, le drone ne doit pas descendre trop bas. Définissez une altitude idéale $z_{\text{désiré}} = \min(2.0, 1.5 \times d_{xy})$.
* `r_shaping = - 1.0 * d_xy - 1.5 * abs(d_z - z_desire) - 0.3 * norm(delta_v)`
* **Bonus d'atterrissage :** Si le contact est détecté ET que $\Vert{}\Delta \mathbf{v}\Vert{} < 0.3\text{ m/s}$ ET que le drone est à plat : donnez `+100` points et coupez ses moteurs ! Si contact mais $\Vert{}\Delta \mathbf{v}\Vert{} > 1.0\text{ m/s}$ (crash violent sur le chariot) : donnez `-50` points.


3. **Jalon 6 : Curriculum Learning & Domain Randomization:** Fichier : phase2_industry/train_landing.py.
Si vous lancez l'entraînement directement avec un chariot qui roule à $3\text{ m/s}$ et du vent, le drone n'apprendra jamais. Voici les deux techniques professionnelles à implémenter :

**1. Le Curriculum Learning (Apprentissage progressif) :**

* Créez une variable `difficulty` dans votre environnement qui va de `0.0` à `1.0`.
* La vitesse maximale du chariot est `v_max = 2.5 * difficulty`.
* Au début de l'entraînement (`difficulty = 0`), le chariot est immobile. Dès que le taux de réussite d'atterrissage dépasse $80\%$ sur les 100 derniers épisodes, augmentez `difficulty` de $+0.1$. Le drone va d'abord apprendre à atterrir à l'arrêt, puis sur une cible lente, puis rapide !

**2. La Domain Randomization (Dans la fonction `reset()`) :**

* Une fois que `difficulty > 0.5`, modifiez les paramètres internes de MuJoCo au début de chaque épisode :
* **Masse du drone :** `model.body_mass[drone_id] = np.random.uniform(0.85, 1.15)`
* **Friction du tapis :** `model.geom_friction[pad_id, 0] = np.random.uniform(0.4, 1.2)` (Tapis glissant vs rugueux)
* **Bruit des capteurs :** Ajoutez un léger bruit gaussien (`np.random.normal(0, 0.01)`) sur l'observation envoyée au réseau.
* **Vent :** Appliquez une force externe aléatoire dans `data.xfrc_applied[drone_id]`.




---

### Récapitulatif de votre boîte à outils logicielle

Pour réaliser l'intégralité de ces 6 jalons sur votre ordinateur (sans GPU), vous n'aurez besoin que d'un environnement virtuel Python contenant 4 bibliothèques :

```bash
pip install numpy matplotlib gymnasium mujoco stable-baselines3

```

* **Astuce de performance sur CPU :** Dans `Stable-Baselines3`, au lieu d'utiliser `DummyVecEnv` (qui fait tourner 1 seul drone), utilisez `SubprocVecEnv` avec 4 ou 8 environnements en parallèle (selon le nombre de cœurs de votre processeur). Cela utilisera $100\%$ de votre CPU et divisera votre temps d'entraînement par 4 ou 8.