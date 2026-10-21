La conception de la fonction de récompense (Reward Shaping) est sans doute la tâche la plus complexe et la plus chronophage pour un ingénieur en Reinforcement Learning. PPO a beau être un excellent algorithme, si vous lui donnez un mauvais signal, il optimisera ce mauvais signal à la perfection.

Voici la fiche complète pour maîtriser le Reward Shaping en production.

---

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