

# Cours Fondamental : Apprentissage par Renforcement sous MuJoCo

L'objectif de cette phase est de créer un "cerveau" (l'Agent RL) capable de stabiliser un système physiquement instable (le drone) et de l'amener sur une cible mobile (le camion). Pour y parvenir, il faut comprendre où s'arrête la simulation physique, et où commence l'intelligence artificielle.

## Module 1 : Ce que MuJoCo gère sous le capot (La Physique)

En tant qu'ingénieur utilisant MuJoCo, vous ne coderez jamais les équations de la physique, mais vous devez savoir ce que le moteur calcule 100 fois par seconde ($100\text{ Hz}$) à chaque appel de `mujoco.mj_step(model, data)`.

**1. Les Repères et l'Attitude (Quaternions)**
MuJoCo gère deux mondes simultanés :

* **Le Repère Monde (Inertiel) :** Fixe, avec la gravité tirant vers le $-Z$.
* **Le Repère Corps (Local) :** Attaché au drone, où les hélices tirent vers le $+Z$ local.
* *Ce que vous devez savoir :* Pour éviter le "Gimbal Lock" (le plantage mathématique si le drone est à la verticale), MuJoCo calcule les rotations avec des **Quaternions** $\mathbf{q} = (w, x, y, z)$. Votre IA ne lira pas des angles classiques (roulis, tangage, lacet), mais des quaternions.

**2. La Dynamique (TDM et PFD)**
Lorsque vous envoyez une puissance aux moteurs, MuJoCo résout instantanément la matrice d'inertie complète du drone. Si vous poussez plus fort à l'arrière, MuJoCo calcule le couple généré, applique le Théorème du Moment Cinétique, et fait piquer le drone du nez. Il applique ensuite le PFD pour calculer l'accélération résultante compte tenu de la gravité.

**3. L'Intégration du Temps**
MuJoCo utilise un solveur implicite d'optimisation (beaucoup plus avancé que la méthode de Runge-Kutta 4) pour calculer la nouvelle position du drone au pas de temps suivant, tout en gérant les chocs élastiques avec la cage invisible sans faire exploser les mathématiques.

---

## Module 2 : L'Interface (Le standard Gymnasium)

Puisque MuJoCo gère la physique, votre seul travail d'ingénieur logiciel est de créer le "câblage" entre MuJoCo et l'IA. C'est le rôle de la classe **Gymnasium** (`drone_env.py`), qui repose sur trois piliers :

**1. L'Espace d'Observation ($S_t$) : Les Yeux de l'IA**
Le réseau de neurones est aveugle. À chaque pas de temps, vous devez extraire les données de MuJoCo (`data.qpos` pour les positions, `data.qvel` pour les vitesses) et les lui donner. Pour atterrir sur le camion, l'IA a besoin de :

* **L'erreur de position :** La distance relative $(X, Y, Z)$ entre le drone et la plateforme du camion.
* **L'erreur de vitesse :** La différence de vitesse entre le drone et le camion.
* **L'attitude :** L'inclinaison actuelle du drone (le quaternion).
* **La vitesse angulaire :** À quelle vitesse le drone est en train de basculer (les gyroscopes).
* *Règle d'or :* Toutes ces valeurs doivent être **normalisées** (idéalement comprises entre $-1$ et $1$). Les réseaux de neurones ne comprennent pas les grandeurs physiques hétérogènes.

**2. L'Espace d'Action ($A_t$) : Les Muscles de l'IA**
L'algorithme IA va recracher un tableau de 4 nombres (pour les 4 moteurs), strictement compris entre $-1.0$ et $1.0$.
C'est à vous d'écrire la ligne de code qui traduit cette décision mathématique en vraie poussée (ex: convertir $[-1, 1]$ en $[0\text{ N}, 5\text{ N}]$) avant de l'envoyer dans `data.actuator("thrust_fl").ctrl`.

**3. Le cycle de vie (`reset` et `step`)**

* **`reset()` :** Remet le drone en l'air, le camion à zéro, et renvoie la première Observation.
* **`step(action)` :** Applique l'Action aux moteurs, fait avancer MuJoCo d'un pas de temps, lit la nouvelle Observation, et calcule la **Récompense**.

---

## Module 3 : Le Cerveau (L'Algorithme PPO)

Nous n'allons pas coder l'IA de zéro. Nous utiliserons **PPO (Proximal Policy Optimization)** via la bibliothèque *Stable-Baselines3*. PPO est l'état de l'art pour le contrôle continu en robotique. Il utilise deux réseaux de neurones en tandem (l'architecture Actor-Critic) :

* **L'Acteur (Le Pilote) :** Il lit l'Observation et bouge les joysticks (les 4 actions).
* **Le Critique (Le Coach) :** Il lit l'Observation et essaie de deviner combien de points le drone va marquer d'ici la fin de l'essai.
* *L'apprentissage :* Si l'Acteur fait une action au hasard qui rapporte **plus** de points que ce que le Critique avait prédit (on appelle ça un *Avantage positif*), l'Acteur renforce les connexions neuronales qui ont mené à cette décision. Si l'action mène au crash, l'Avantage est négatif, le comportement est "effacé".

---

## Module 4 : L'Art du Dressage (Le Reward Shaping)

C'est ici que se trouve 90% de votre futur travail d'ingénieur en IA. Vous devez concevoir la formule mathématique de la Récompense ($R_t$). L'IA étant une tricheuse professionnelle (Reward Hacking), une bonne fonction de récompense s'articule toujours en plusieurs couches :

**1. Les Récompenses Continues (Dense Rewards) :** Distribuées à chaque fraction de seconde pour guider le drone.

* *Survie :* Un léger bonus ($+0.1$) à chaque pas de temps passé en l'air.
* *Attraction gravitationnelle :* Une pénalité mathématique proportionnelle à la distance avec le camion. Plus il s'éloigne, plus il perd de points. Cela le force à se rapprocher.
* *Stabilité :* Une pénalité s'il s'incline trop, pour l'obliger à voler à plat.
* *Économie d'énergie :* Une micro-pénalité sur l'usage des moteurs, pour éviter qu'il ne fasse osciller ses hélices violemment à 100%.

**2. Les Récompenses Terminales (Sparse Rewards) :** Distribuées uniquement à la fin d'un essai.

* *Le Crash :* Si le drone touche le sol ou se retourne, $-100$ points, et l'épisode s'arrête (fonction `step()` renvoie `terminated = True`).
* *Le Succès :* S'il touche la boîte de collision de la zone d'atterrissage à une vitesse raisonnable, $+1000$ points, fin de l'épisode.

