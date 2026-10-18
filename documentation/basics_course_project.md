Faire atterrir un drone de manière autonome sur une cible mobile regroupe quatre grands domaines : **la géométrie dans l'espace (les repères)**, **la mécanique du vol (les forces)**, **les mathématiques de simulation (le temps)** et **l'apprentissage par renforcement (le cerveau)**.

Voici le cours complet, repris depuis zéro, pour comprendre exactement ce qui se passe sous le capot de votre futur code.

---

## Module 1 : Se repérer dans l'espace 3D (Cinématique)

Avant de parler de forces ou d'intelligence artificielle, il faut savoir décrire où se trouve le drone et comment il est orienté. C'est le piège numéro un en robotique : confondre le point de vue du drone et celui du spectateur.

### 1. Les deux repères indispensables

Imaginez que vous êtes debout dans un champ et que vous regardez votre drone voler. Il existe deux mondes simultanés :

* **Le Repère Monde (ou Inertiel), noté $\mathcal{R}_W$ :** C'est votre point de vue fixe au sol. L'axe $X$ pointe vers le Nord, l'axe $Y$ vers l'Ouest, et l'axe $Z$ pointe droit vers le ciel. **La gravité tire toujours vers le $-Z$ de ce repère**, peu importe comment le drone est tourné.
* **Le Repère Corps (ou Mobile), noté $\mathcal{R}_B$ (pour *Body*) :** C'est le point de vue d'une fourmi assise au centre du drone. L'axe $X_B$ pointe vers le nez du drone, $Y_B$ vers son flanc gauche, et $Z_B$ sort par le dessus de sa coque. **Les hélices poussent toujours vers le $+Z_B$ de ce repère**, car elles sont vissées sur le châssis.

---

Comme l'illustre le schéma ci-dessus, le drone peut tourner autour de ses trois axes locaux ($X, Y, Z$) :

* **Le Roulis (Roll) :** Il penche à gauche ou à droite (rotation autour de l'axe avant-arrière $X$).
* **Le Tangage (Pitch) :** Il penche le nez vers le bas ou vers le haut (rotation autour de l'axe gauche-droite $Y$).
* **Le Lacet (Yaw) :** Il tourne sur lui-même comme une toupie à plat (rotation autour de l'axe vertical $Z$).

### 2. Pourquoi un drone n'avance-t-il que s'il se penche ?

Un drone standard n'a pas de réacteur à l'arrière pour le pousser vers l'avant. Ses 4 hélices pointent uniquement vers le haut de sa coque ($+Z_B$).

* S'il reste parfaitement à plat, la poussée des hélices pointe à 100% vers le ciel ($+Z$ du monde) : il monte ou descend verticalement.
* Pour avancer vers le Nord, il doit activer le **tangage** (pencher le nez vers l'avant). Ainsi, la flèche de poussée de ses hélices est inclinée : une partie de la force continue de le maintenir en l'air contre la gravité, et une partie le tire vers l'avant.

### 3. La Matrice de Rotation et les Quaternions

Pour que votre programme Python sache dans quelle direction du Monde ($\mathcal{R}_W$) pointent les hélices du Drone ($\mathcal{R}_B$), il faut un traducteur mathématique : **la Matrice de Rotation $R$** (un tableau de $3 \times 3$ chiffres).
Si vous multipliez un vecteur vu par la fourmi sur le drone par $R$, vous obtenez ce même vecteur vu depuis le sol :

$$\mathbf{F}_{\text{monde}} = R \cdot \mathbf{F}_{\text{corps}}$$

**Pourquoi parle-t-on de Quaternions ?**
Pour construire cette matrice $R$, l'être humain aime utiliser les 3 angles (Roulis, Tangage, Lacet). Mais mathématiquement, si le drone se cabre à 90° exactement à la verticale, deux axes se superposent et les formules divisent par zéro : c'est le *Gimbal Lock* (blocage de cardan), votre simulation plante.
Pour éviter cela, l'ordinateur stocke l'orientation sous forme d'un **Quaternion** $\mathbf{q} = (q_w, q_x, q_y, q_z)$. Voyez cela simplement comme un code à 4 chiffres immunisé contre les bugs qui définit une orientation 3D unique, à partir duquel on calcule proprement la matrice $R$ sans jamais planter.

---

## Module 2 : La Mécanique du Vol (Comment les moteurs créent le mouvement)

Votre drone a 4 moteurs mais il doit contrôler 6 mouvements (avancer/reculer, gauche/droite, monter/descendre, et les 3 rotations). Comment fait-il ?

### 1. Le Mixage Moteur (Motor Mixing)

Chaque moteur $i$ (numérotés de 1 à 4) tourne à une certaine vitesse et génère une force de poussée vers le haut $f_i$ (en Newtons).

| Mouvement souhaité | Action sur les 4 moteurs | Effet physique |
| --- | --- | --- |
| **Monter / Descendre** | Augmenter ou baisser les 4 moteurs ensemble. | Change la poussée totale $F_z = f_1 + f_2 + f_3 + f_4$. |
| **Tangage (Avancer)** | Pousser plus fort à l'arrière qu'à l'avant. | Crée un effet de levier (couple $\tau_y$) qui fait piquer du nez. |
| **Roulis (Aller à droite)** | Pousser plus fort à gauche qu'à droite. | Crée un effet de levier (couple $\tau_x$) qui penche à droite. |
| **Lacet (Pivoter à plat)** | Accélérer 2 moteurs en diagonale, ralentir les 2 autres. | Utilise la résistance de l'air sur les pales (couple réactif $\tau_z$). |

*Point curieux sur le Lacet :* Si les 4 hélices tournaient dans le même sens, le corps du drone tournerait en sens inverse comme une toupie folle (par conservation de l'élan, comme un hélicoptère qui a perdu son rotor de queue). C'est pourquoi sur un drone, **2 hélices en diagonale tournent dans le sens horaire, et les 2 autres dans le sens anti-horaire**. En vol normal, leurs effets de toupie s'annulent !

---

### 2. Le Principe Fondamental de la Dynamique (PFD) — *Pour la Translation*

Le PFD (2ème loi de Newton) sert à savoir comment le centre de gravité du drone se déplace dans le champ (Repère Monde). La formule dit : **Masse $\times$ Accélération = Somme des forces**.

$$m \mathbf{a} = \mathbf{F}_{\text{gravité}} + \mathbf{F}_{\text{poussée\_monde}}$$

1. **La gravité :** Tire vers le bas avec une force $\mathbf{F}_{\text{gravité}} = \begin{pmatrix} 0 \\ 0 \\ -m g \end{pmatrix}$.
2. **La poussée :** Dans le repère du drone, les hélices poussent droit vers le haut : $\mathbf{F}_{\text{corps}} = \begin{pmatrix} 0 \\ 0 \\ F_z \end{pmatrix}$. On utilise notre traducteur (la matrice $R$) pour savoir ce que ça donne dans le monde : $\mathbf{F}_{\text{poussée\_monde}} = R \cdot \mathbf{F}_{\text{corps}}$.

En divisant par la masse $m$, vous obtenez l'accélération $\mathbf{a} = (a_x, a_y, a_z)$ du drone dans les airs !

---

### 3. Le Théorème du Moment Cinétique (TDM) — *Pour la Rotation*

Le TDM est l'exact équivalent du PFD, mais pour les rotations. Il explique comment les effets de levier des moteurs (les couples $\boldsymbol{\tau} = (\tau_x, \tau_y, \tau_z)$) font accélérer la vitesse de rotation du drone $\boldsymbol{\omega} = (p, q, r)$.

En version simplifiée, on pourrait croire que : $\text{Inertie} \times \text{Accélération Angulaire} = \text{Couples}$. Mais en 3D, un objet qui tourne subit un effet gyroscopique (comme une roue de vélo qu'on essaie de pencher pendant qu'elle tourne). La vraie formule d'Euler dans le repère du drone est :

$$I \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (I \boldsymbol{\omega}) = \boldsymbol{\tau}$$

* **$I$ (La Matrice d'Inertie) :** C'est l'équivalent de la masse, mais pour la rotation. Plus le drone est large et lourd aux extrémités, plus $I$ est grand, et plus il est difficile de le faire tourner vite. Comme un drone est symétrique, c'est juste 3 chiffres $(I_{xx}, I_{yy}, I_{zz})$ sur la diagonale d'un tableau $3 \times 3$.
* **$\dot{\boldsymbol{\omega}}$ :** C'est l'accélération angulaire que l'on cherche à calculer.
* **$\boldsymbol{\omega} \times (I \boldsymbol{\omega})$ :** Le produit vectoriel qui représente la résistance gyroscopique.

En isolant $\dot{\boldsymbol{\omega}}$, on obtient :

$$\dot{\boldsymbol{\omega}} = I^{-1} \left( \boldsymbol{\tau} - \boldsymbol{\omega} \times (I \boldsymbol{\omega}) \right)$$

---

## Module 3 : Faire avancer le temps (L'Intégration Numérique)

Dans la vraie vie, le temps coule de manière continue. Dans un ordinateur, le temps avance par petits sauts discrets, comme les images d'un film : toutes les $dt = 0.01\text{ seconde}$ (100 Hz).

Grâce au PFD et au TDM au Module 2, vous connaissez à l'instant $t$ l'accélération linéaire $\mathbf{a}$ et l'accélération angulaire $\dot{\boldsymbol{\omega}}$. Comment deviner où sera le drone à l'instant $t + dt$ ? C'est **l'intégration numérique**.

* **La méthode naïve (Euler) :** On suppose que la vitesse et l'accélération restent constantes pendant les $0.01\text{s}$ :
* $\text{Nouvelle Vitesse} = \text{Ancienne Vitesse} + \text{Accélération} \times dt$
* $\text{Nouvelle Position} = \text{Ancienne Position} + \text{Ancienne Vitesse} \times dt$
* *Problème :* En rotation 3D, le drone tourne pendant le pas de temps $dt$. Calculer en ligne droite accumule de minuscules erreurs qui font "exploser" la simulation après quelques secondes.


* **La méthode pro (Runge-Kutta 4 ou RK4) :** Au lieu de regarder uniquement la pente au début du saut, RK4 "sonde" la physique 4 fois (au début, deux fois au milieu du saut de temps, et à la fin) et fait une moyenne pondérée. C'est une simple recette de cuisine de 6 lignes de code en Python qui rend votre physique 3D ultra-précise et incassable.

---

## Module 4 : Le Reinforcement Learning (Le Cerveau)

Maintenant que vous avez un drone virtuel qui obéit aux lois de la physique, il tombe comme une pierre si personne ne contrôle ses 4 moteurs. Au lieu de coder un pilote automatique classique avec des équations compliquées (PID), vous allez utiliser l'apprentissage par renforcement (RL).

### 1. La boucle fondamentale du RL

Le principe est identique au dressage d'un chien : l'essai-erreur guidé par des friandises.

---

Tout votre code RL tourne autour de cette boucle infinie (souvent gérée par la bibliothèque `Gymnasium`) :

1. **L'Environnement (`Env`) :** C'est le monde physique que vous avez codé aux Modules 1, 2 et 3 (le drone + le chariot qui bouge au sol).
2. **L'Agent (Le Réseau de Neurones) :** C'est le pilote. Au début de l'entraînement, son cerveau est rempli de nombres aléatoires : il ne sait rien faire et met des gaz au hasard.
3. **L'État ou Observation ($S_t$) :** À chaque instant $t$ (toutes les $0.01\text{s}$), l'environnement envoie un bulletin d'information à l'Agent. Pour atterrir sur le chariot, que doit savoir le pilote ?
* La distance relative au chariot : $(\Delta x, \Delta y, \Delta z) = \mathbf{p}_{\text{drone}} - \mathbf{p}_{\text{chariot}}$.
* La vitesse relative : $(\Delta v_x, \Delta v_y, \Delta v_z) = \mathbf{v}_{\text{drone}} - \mathbf{v}_{\text{chariot}}$.
* Son inclinaison actuelle (le quaternion $\mathbf{q}$ ou la matrice $R$) et sa vitesse de rotation $(p, q, r)$.


4. **L'Action ($A_t$) :** En lisant cet État, le réseau de neurones fait des multiplications matricielles et recrache 4 chiffres : la puissance à envoyer aux 4 moteurs.
5. **La Récompense ($R_t$ - *Reward*) :** L'environnement avance de $0.01\text{s}$ avec RK4 et attribue une note en points à l'Agent pour lui dire si son action a amélioré ou empiré la situation.

### 2. Comment le réseau apprend-il vraiment ? (L'architecture Actor-Critic / PPO)

Comme les 4 moteurs acceptent des valeurs continues (ex: $42.5\%$ de puissance, pas juste "Allumé/Éteint"), on utilise des algorithmes comme **PPO (Proximal Policy Optimization)**. En interne, PPO utilise deux petits réseaux de neurones qui travaillent en binôme :

* **L'Acteur (Actor - La Politique $\pi$) :** C'est celui qui tient les manettes. Il prend l'État en entrée et décide de l'Action (les 4 poussées moteurs).
* **Le Critique (Critic - La Fonction de Valeur $V$) :** C'est le coach assis sur le siège passager. Il regarde l'État et essaie de prédire : *"Vu notre position et notre vitesse actuelles, je parie qu'on va gagner 50 points d'ici la fin de l'épisode"*.

**Le déclic de l'apprentissage :**
Si l'Acteur tente une manœuvre par hasard (ex: il incline le drone vers le chariot) et obtient finalement **80 points** alors que le Critique n'en attendait que **50**, c'est une **bonne surprise** (l'*Avantage* est positif : $+30$). L'algorithme modifie alors les connexions neuronales de l'Acteur pour qu'il reproduise ce geste plus souvent dans le futur ! Si la manœuvre mène au crash ($-100$ points), l'Avantage est négatif et l'Acteur apprend à ne plus jamais faire ça.

### 3. L'Art de la Fonction de Récompense (Reward Shaping)

C'est vous qui écrivez la règle du jeu. Si votre règle est mal écrite, l'IA va tricher (c'est le *Reward Hacking*).

* *Mauvais exemple :* Vous donnez $+100$ points uniquement quand le drone touche le chariot, et $0$ le reste du temps. Comme le drone bouge au hasard au début, il a une chance sur un milliard de toucher le chariot qui bouge. Il n'aura jamais les $+100$ points et n'apprendra jamais rien.
* *Autre piège :* Vous lui donnez $-1$ point par mètre de distance avec le chariot à chaque milliseconde, sans pénaliser le crash. Le drone va comprendre que s'il fonce tête la première dans le sol pour se crasher en $0.1\text{s}$, l'épisode s'arrête tout de suite et il perd moins de points que s'il vole longtemps en essayant d'atterrir !
* *La bonne formule :* On combine un petit bonus de survie à chaque pas de temps, une récompense qui augmente au fur et à mesure que la distance $\Vert{}\Delta \mathbf{p}\Vert{}$ diminue, et une pénalité si la vitesse d'approche est trop violente près du sol.

---

## Expérimentez les concepts par vous-même

Pour ancrer ce que nous venons de voir aux Modules 1 et 2, utilisez le simulateur interactif ci-dessous. Vous pouvez y manipuler les commandes de poussée et de couple pour observer comment la matrice de rotation incline le vecteur de poussée total (flèche bleue) par rapport à la gravité (flèche rouge) et comment se crée l'accélération en 3D :

---

> **À retenir de cette visualisation :** Remarquez que même lorsque le drone penche fortement (Roulis ou Tangage), la flèche de poussée (liée au drone) penche avec lui, tandis que la flèche de gravité pointe obstinément droit vers le sol. C'est l'addition de ces deux flèches qui crée la flèche d'accélération horizontale permettant de rattraper le chariot.