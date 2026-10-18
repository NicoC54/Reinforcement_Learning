Pour éviter le syndrome de la "page blanche" ou de vous noyer dans les équations et le débogage RL simultanément, la règle d'or d'un ingénieur roboticien est d'**isoler les problèmes**. Ne touchez pas au Reinforcement Learning tant que votre physique n'est pas certifiée à 100%.

Voici votre feuille de route découpée en **5 jalons (Sprints) indépendants**. Chaque étape a un critère de validation précis pour vous prouver que votre code fonctionne avant de passer à la suite.

---

1. **Le Moteur Physique Pur (PFD, TDM et Cinématique 3D):** Objectif : Réviser la mécanique du solide sans aucune IA.
Créez un simple script Python (`physics.py`) avec une classe `Drone3D`. Ne pensez ni au chariot, ni à Gymnasium pour l'instant.

* **Le vecteur d'état ($X$) :** Définissez un vecteur NumPy de 13 valeurs : Position $(x, y, z)$, Vitesse linéaire $(v_x, v_y, v_z)$, Orientation par Quaternion $(q_w, q_x, q_y, q_z)$, et Vitesse angulaire dans le repère du corps $(p, q, r)$.
* **Le Mixage Moteur :** Écrivez la matrice qui transforme vos 4 commandes moteurs $(u_1, u_2, u_3, u_4)$ en une poussée verticale totale $F_z$ et trois couples $(\tau_x, \tau_y, \tau_z)$ selon la géométrie en "X" ou en "+" de votre drone.
* **Les Équations du Mouvement :**
* **PFD (Repère Inertiel Monde) :** $\dot{\mathbf{v}} = \frac{1}{m} R(\mathbf{q}) \begin{pmatrix} 0 \\ 0 \\ F_z \end{pmatrix} - \begin{pmatrix} 0 \\ 0 \\ g \end{pmatrix}$ (où $R(\mathbf{q})$ est la matrice de rotation issue de votre quaternion).
* **TDM (Repère Corps Mobile) :** $\dot{\boldsymbol{\omega}} = I^{-1} (\boldsymbol{\tau} - \boldsymbol{\omega} \times (I \boldsymbol{\omega}))$.
* **Cinématique des Quaternions :** $\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{pmatrix} 0 \\ \boldsymbol{\omega} \end{pmatrix}$. N'oubliez pas de re-normaliser $\mathbf{q}$ à chaque pas ($\Vert{}\mathbf{q}\Vert{} = 1$).


* **Intégration :** Codez un intégrateur **Runge-Kutta d'ordre 4 (RK4)** plutôt qu'Euler explicite. En 3D, Euler fait diverger les quaternions trop vite.

**Test de validation (Sanity Check) :**

1. Mettez les 4 moteurs à $0$ : le drone doit tomber en chute libre ($z(t) = z_0 - \frac{1}{2}gt^2$) sans tourner.
2. Mettez chaque moteur exactement à $\frac{mg}{4}$ : l'accélération verticale doit être de $0.00000$.
3. Augmentez légèrement le moteur avant-droit : vérifiez le signe de $\tau_x, \tau_y, \tau_z$ et le sens de rotation.


2. **Visualisation 3D Minimaliste:** Objectif : Voir ce que vous faites pour débugger facilement.
On ne peut pas débugger de la robotique 3D en lisant des chiffres dans un terminal. Créez un visualiseur léger avec `Matplotlib` (module `mpl_toolkits.mplot3d`).

* Représentez le drone par une croix en 3D (2 segments qui se croisent au centre de masse). Utilisez votre matrice de rotation $R(\mathbf{q})$ pour orienter les bras de la croix dans le repère global.
* Dessinez un carré au sol ($z=0$) pour figurer le chariot.

**Test de validation :** Lancez une boucle `for` qui applique une poussée légèrement asymétrique et affichez l'animation. Vous devez voir votre croix décoller, s'incliner et se retourner de manière fluide.


3. **Encapsulation Gymnasium et Premier Vol (Hovering):** Objectif : Réviser les bases du RL sur un problème simple.
Maintenant que la physique est saine, encapsulez-la dans le standard RL (`env.py`).

* **Structure :** Héritez de `gymnasium.Env`. Implémentez `__init__`, `reset()`, et `step(action)`.
* **Action Space :** `Box(low=-1, high=1, shape=(4,))`. *Astuce d'ingénieur :* les réseaux de neurones préfèrent sortir des valeurs entre $-1$ et $1$. Transformez ensuite cet intervalle dans votre `step()` vers $[0, F_{\max}]$.
* **Observation Space :** Donnez-lui l'erreur de position par rapport à une cible fixe $(0, 0, 2\text{m})$, sa vitesse, sa matrice de rotation (ou quaternion) et sa vitesse angulaire. Normalisez les entrées !
* **Reward simple :** $R = \text{bonus\_survie} - \Vert{}\text{erreur\_pos}\Vert{} - 0.1 \Vert{}\mathbf{v}\Vert{} - 0.1 \Vert{}\boldsymbol{\omega}\Vert{}$.
* **Terminaison (`terminated`) :** Si le drone sort d'une boîte virtuelle (ex: $> 5\text{m}$) ou s'il se retourne complètement ($R_{zz} \text{ inférieur à } 0$), coupez l'épisode pour ne pas gaspiller de temps de calcul.

**Test de validation :** Lancez l'algorithme **PPO** de `Stable-Baselines3` pendant 300 000 pas (environ 5 à 10 minutes sur CPU). Le drone doit apprendre à se stabiliser au point fixe $(0,0,2)$.


4. **Cinématique du Chariot et Repère Relatif:** Objectif : Introduire la cible mouvante et travailler les changements de repère.
Intégrez le chariot dans votre simulation sans toucher aux équations du drone.

* **Modèle du chariot :** Pas besoin de PFD pour le chariot, la cinématique suffit. Faites-le évoluer dans le plan $(x, y, z=0)$. Pour un mouvement aléatoire réaliste (non téléporté), faites varier son **accélération** aléatoirement (processus d'Ornstein-Uhlenbeck ou bruit gaussien lissé), puis intégrez pour avoir sa vitesse $\mathbf{v}_c$ et sa position $\mathbf{p}_c$.
* **Changement de l'Observation Space (Crucial) :** Le drone n'a pas besoin de connaître sa position absolue dans le monde, mais sa **position et vitesse relatives** au chariot : $\Delta \mathbf{p} = \mathbf{p}_{\text{drone}} - \mathbf{p}_{\text{chariot}}$ et $\Delta \mathbf{v} = \mathbf{v}_{\text{drone}} - \mathbf{v}_{\text{chariot}}$.

**Test de validation :** Entraînez le drone à faire du *tracking* (suivre le chariot en restant exactement $1\text{m}$ au-dessus de lui, sans atterrir).


5. **Atterrissage Dynamique et Reward Shaping:** Objectif : Résoudre le problème complet.
C'est l'étape finale où vos compétences RL seront mises à l'épreuve. Atterrir demande d'aller vers $z=0$ tout en ayant une vitesse relative quasi nulle au moment de l'impact.

* **Le Cône d'Approche :** Modifiez la récompense pour forcer le drone à descendre uniquement s'il est bien aligné en $(x, y)$ avec le chariot. S'il est loin horizontalement, il doit rester en altitude.
* **Conditions de succès :** Si $z \le z_{\text{chariot}}$, que la distance horizontale est dans le rayon du chariot, que la vitesse d'impact $\Vert{}\Delta \mathbf{v}\Vert{}$ est douce et que le drone est à plat : donnez un gros bonus (ex: $+100$) et terminez l'épisode avec succès.
* **Curriculum Learning (Optionnel mais très formateur) :** Commencez l'entraînement avec un chariot immobile, puis augmentez progressivement la vitesse maximale du chariot au fil des épisodes.

**Test de validation :** Votre drone poursuit le chariot, synchronise sa vitesse horizontale, descend progressivement et coupe ses gaz au contact.


---

### L'architecture de fichiers recommandée

Pour garder les idées claires, séparez strictement votre code en 4 fichiers dès le départ :

| Fichier | Rôle unique | Dépendances principales |
| --- | --- | --- |
| `drone_physics.py` | Contient les constantes ($m, g, I$), le PFD, le TDM et l'intégrateur RK4. | `numpy` |
| `visualizer.py` | Gère l'affichage 3D de la croix et du chariot. | `matplotlib` |
| `landing_env.py` | L'environnement RL qui connecte la physique, le chariot et calcule la récompense. | `gymnasium`, `numpy` |
| `train.py` | Le script de lancement de l'entraînement PPO/SAC et d'évaluation du modèle. | `stable_baselines3` |