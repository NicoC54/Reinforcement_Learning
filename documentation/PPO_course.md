
PPO (Proximal Policy Optimization) est aujourd'hui le standard de l'industrie en apprentissage par renforcement (c'est l'algorithme derrière l'entraînement RLHF de ChatGPT). Son succès repose sur un équilibre parfait : il est presque aussi stable que des méthodes mathématiquement complexes (comme TRPO) tout en étant aussi simple et rapide à exécuter que des méthodes basiques.

Voici le guide orienté ingénierie pour comprendre, configurer et déboguer PPO en production.

## 1. La Théorie : L'intuition derrière la mécanique

PPO appartient à la famille des algorithmes **Actor-Critic** basés sur le **Policy Gradient**.

* **L'Acteur (Policy Network) :** Décide quelle action prendre dans un état donné.
* **Le Critique (Value Network) :** Évalue la qualité de l'état actuel (prédit la somme des récompenses futures).

Le problème majeur des anciens algorithmes de Policy Gradient était la **stabilité**. Si une action donnait une excellente récompense par hasard, l'algorithme mettait à jour ses poids de manière trop brutale dans cette direction, "cassant" la politique qui fonctionnait bien jusque-là.

**L'innovation de PPO : Le "Clipping" (Écrêtage)**
PPO empêche la politique de changer de manière trop drastique lors d'une seule mise à jour. Il calcule un ratio de probabilité entre la nouvelle politique et l'ancienne :
$r_t(\theta) = \frac{\pi_\theta(a_t\vert{}s_t)}{\pi_{old}(a_t\vert{}s_t)}$

Au lieu d'optimiser ce ratio sans limite, PPO le "clip" (le contraint) dans un intervalle, généralement $[0.8, 1.2]$.

* Si une action devient beaucoup plus probable qu'avant (ratio > 1.2) et qu'elle a un bon score, on arrête de l'encourager pour ce tour.
* Cela garantit une **mise à jour pessimiste et prudente**, évitant l'effondrement des performances.

---

## 2. Les Subtilités Pratiques (Ce qui fait marcher PPO)

Pour qu'un modèle PPO fonctionne dans la vraie vie, l'équation de base ne suffit pas. Les implémentations (comme Stable-Baselines3 ou Ray RLlib) intègrent des mécanismes cruciaux :

* **L'Avantage (GAE - Generalized Advantage Estimation) :** PPO n'utilise pas la récompense brute pour se mettre à jour. Il utilise "l'avantage". L'avantage répond à la question : *Cette action a-t-elle été meilleure que ce que le Critique avait prédit pour cet état ?* Le GAE permet de lisser ces estimations pour réduire la variance du gradient.
* **Mises à jour par Epochs :** Contrairement aux anciens algorithmes qui jetaient les données après un pas, PPO stocke un grand *batch* de transitions (ex: 2048 pas), puis effectue plusieurs passages d'optimisation (epochs) sur ces mêmes données en les divisant en *minibatches*. C'est ce qui le rend efficace en termes d'échantillons (Sample Efficient).
* **Normalisation des Observations et des Avantages :** En pratique, alimenter un réseau de neurones avec des observations non normalisées détruit la stabilité de PPO. L'implémentation standard maintient une moyenne et une variance glissantes pour normaliser les *inputs* et les *avantages* à chaque étape.

---

## 3. Guide de Réglage des Hyperparamètres

C'est ici que se trouve le vrai travail d'ingénierie. PPO est robuste, mais un mauvais paramétrage le rendra aveugle ou instable.

| Paramètre | Valeur Standard | Rôle et impact pratique |
| --- | --- | --- |
| **`learning_rate`** | $3 \times 10^{-4}$ | Taux d'apprentissage d'Adam. Si l'apprentissage s'effondre soudainement, réduisez-le ou utilisez un *learning rate schedule* linéaire (qui décroît vers 0). |
| **`n_steps` (Rollout)** | 2048 | Nombre d'étapes collectées par environnement avant de faire une mise à jour. Doit être suffisamment grand pour capturer de vrais signaux de récompense (souvent la taille de l'horizon de l'épisode). |
| **`batch_size`** | 64 à 256 | Taille du minibatch lors de l'entraînement du réseau. Un batch trop petit rend le gradient bruyant. Pour des environnements continus complexes, montez à 512 ou 1024. |
| **`n_epochs`** | 10 | Nombre de fois que le réseau s'entraîne sur le même *rollout*. Trop élevé = surapprentissage du batch actuel (la politique tourne en rond). Trop bas = apprentissage trop lent. |
| **`gamma` ($\gamma$)** | 0.99 | Facteur de discount. Définit à quel point le modèle se soucie du long terme. Pour des épisodes très longs, montez à 0.999. Pour des jeux très courts/réactifs, descendez à 0.95. |
| **`gae_lambda` ($\lambda$)** | 0.95 | Équilibre entre le biais et la variance de l'estimation de l'avantage. Une valeur de 1.0 fait confiance aux vraies récompenses (haute variance), 0.8 fait confiance au Critique (haut biais). |
| **`clip_range` ($\epsilon$)** | 0.2 | L'intervalle de clipping $[0.8, 1.2]$. Ne le modifiez presque jamais, sauf si vous réduisez le learning rate (vous pouvez réduire le clip_range à 0.1 en fin d'entraînement). |
| **`ent_coef`** | 0.0 à 0.01 | **Crucial pour l'exploration.** Force l'Acteur à ne pas être 100% sûr de ses actions. Si votre agent converge trop vite vers un comportement sous-optimal (optimum local), augmentez cette valeur. |

---

## 4. Checklist de Débogage de Modèle

En tant qu'ingénieur, vous allez passer plus de temps à regarder les courbes Tensorboard qu'à lire du code. Voici comment diagnostiquer les problèmes courants :

1. **L'entropie (Entropy Loss) chute brutalement à 0 :**
* *Le problème :* L'agent est devenu hyper-déterministe trop tôt et a arrêté d'explorer. Il a trouvé un "hack" ou une sous-performance acceptable.
* *La solution :* Augmentez `ent_coef` (ex: de 0.001 à 0.01), ou ajoutez du bruit dans votre environnement.


2. **La Value Loss (Critic Loss) explose ou ne descend pas :**
* *Le problème :* L'environnement est trop chaotique, ou vos récompenses (*rewards*) ont une échelle absurde (ex: vous donnez +10000 pour un but et -1 pour un pas).
* *La solution :* Divisez vos récompenses par un scalaire pour qu'elles restent globalement dans l'intervalle $[-1, 1]$ ou $[-10, 10]$ par étape.


3. **L'agent ne progresse pas du tout (Reward plate curve) :**
* *Le problème :* Soit le *learning rate* est trop haut (catastrophic forgetting permanent), soit l'agent ne rencontre jamais de signal positif (problème de "Sparse Reward").
* *La solution :* Vérifiez si la tâche est réalisable par des actions aléatoires en début d'entraînement. Si ce n'est pas le cas, implémentez du *Reward Shaping* (donnez des petites récompenses pour s'approcher de l'objectif).

## 5. Le "Reward Shaping" : L'art de guider l'agent

En RL, il existe deux types de récompenses :

* **Sparse (Clairsemée) :** Vous donnez **+100** si l'agent gagne et **0** le reste du temps. C'est ce qu'on veut idéalement, mais c'est presque impossible à apprendre pour PPO au début, car la probabilité qu'un agent trouve la solution au hasard (pour avoir son premier signal positif) est quasi nulle.
* **Dense :** Vous donnez un signal à chaque étape pour indiquer à l'agent s'il "chauffe" ou s'il "refroidit". Le *Reward Shaping* consiste à transformer une récompense sparse en récompense dense.

### Le Fléau du "Reward Hacking" (Spécification Trompeuse)

Les agents PPO sont des tricheurs nés. Ils trouvent le chemin de moindre résistance pour maximiser la récompense mathématique, ignorant complètement l'intention humaine.

* *L'intention :* Apprendre à un robot aspirateur à nettoyer la pièce. Vous lui donnez **+1** pour chaque poussière aspirée.
* *Le Reward Hack :* Le robot aspire la poussière, la recrache, puis la ré-aspire en boucle pour obtenir des points infinis, sans jamais nettoyer le reste de la pièce.

### Les 4 Règles d'Or de l'Ingénieur pour PPO

Pour éviter les comportements aberrants et stabiliser le *Value Network* de PPO, suivez ces principes structurels :

1. **Récompensez le delta (le progrès), pas la position absolue**
Ne donnez jamais une récompense continue juste parce que l'agent "est proche" de l'objectif, sinon il va s'arrêter près de l'objectif pour accumuler des points à l'infini (phénomène de *camping*).
**La règle :** Calculez la distance à l'objectif au temps $t$, puis au temps $t+1$. La récompense doit être la *différence* entre ces deux distances. S'il recule, la récompense devient négative.
2. **Échelle et Normalisation (Crucial pour PPO)**
Le Critique de PPO (Value Network) doit prédire la somme des récompenses futures. Si vos récompenses varient entre 0 et 100 000, le réseau de neurones va diverger.
**La règle :** Calibrez votre fonction pour que la récompense accumulée sur un épisode complet tourne idéalement autour de **0 à 100** (ou que la récompense par étape (step) soit proche de l'intervalle **[-1, 1]**).
3. **Pénalité de Temps (Time Penalty)**
Si vous voulez que votre agent accomplisse la tâche rapidement, ajoutez une légère pénalité à chaque étape (ex: **-0.01**). L'agent comprendra que flâner ou faire des boucles lui coûte des points et cherchera à terminer l'épisode au plus vite.
4. **Early Stopping (Arrêt Prématuré)**
Ne laissez pas l'agent explorer des états irrécupérables. S'il sort des limites de la carte ou se retourne, mettez fin à l'épisode immédiatement avec une forte pénalité (ex: **-10** ou **-100**).
**Pourquoi ?** PPO apprend par "batchs" (Rollouts). Si un épisode dure 2048 étapes alors que l'agent est bloqué dans un mur depuis l'étape 10, vous gaspillez 2038 étapes de calcul qui pollueront l'entraînement.

---

### Étude de cas : Entraîner une voiture autonome à se garer

Voici comment l'ingénierie du Reward Shaping évolue face aux comportements de l'agent.

**V1 : Naïve (Échec garanti)**

* +1000 si elle est garée.
* *Résultat :* La voiture tourne en rond au hasard, ne trouve jamais la place, l'apprentissage reste plat.

**V2 : Dense mais dangereuse (Reward Hack)**

* +1 si elle avance vers la place.
* +1000 si elle est garée.
* *Résultat :* La voiture avance vers la place, fait marche arrière, puis ré-avance pour accumuler des "points d'avancement" infinis sans jamais se garer.

**V3 : Ingénierie PPO Robuste (Succès)**

* **$R_{progres}$ :** (Distance précédente - Distance actuelle) $\times$ 10. *(Si elle avance de 1 mètre, elle gagne +10. Si elle recule d'1 mètre, elle prend -10. Impossible de tricher).*
* **$R_{temps}$ :** -0.1 par étape. *(L'incite à se garer vite).*
* **$R_{collision}$ :** -100 **ET** fin de l'épisode. *(Ne perd pas de temps à simuler une voiture détruite).*
* **$R_{succes}$ :** +100 **ET** fin de l'épisode.

> **Astuce de Pro :** Une fois que votre agent arrive à accomplir la tâche avec une fonction de récompense complexe (V3), vous pouvez progressivement réduire les récompenses intermédiaires (le *shaping*) pour ne garder que la récompense finale. C'est ce qu'on appelle le **Curriculum Learning**.