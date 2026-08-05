# Objet de l’article

L’article étudie le partage opportuniste d’infrastructures RAN entre
opérateurs mobiles, à l’aide de l’optimisation combinatoire et de la
théorie des jeux coopératifs.

# Décisions éditoriales et scientifiques au 5 août 2026

- Cible principale : IEEE Transactions on Green Communications and
  Networking (TGCN).
- Langue finale : anglais ; traduction seulement après stabilisation du fond.
- Statut empirique : étude semi-empirique fondée sur des mesures Orange
  d’Île-de-France, sans observation de quatre opérateurs colocalisés.
- Message central : l’efficacité énergétique de la coopération ne garantit
  pas sa stabilité coalitionnelle.
- Partage : valeur de Shapley si elle appartient au cœur, nucléole sinon.
- Projection de Shapley : diagnostic secondaire.
- Veille : puissance supposée nulle ; \(F_i\) est calibré directement à partir
  de l'intercept actif.
- Fenêtre numérique : bornes horaires paramétrables, à fixer ultérieurement à
  partir du trafic et avant l’analyse de stabilité.
- Politique opérationnelle principale : même ensemble de gardiens sur toute
  la fenêtre, allocations de trafic horaires.
- Capacité : paramètre de scénario construit à partir du pic observé et d’un
  taux de charge au pic explicite ; aucune capacité physique n’est prétendue
  observée.

Le protocole détaillé est fixé dans `NUMERICAL_PROTOCOL.md`.

# Notations stabilisées

- N : ensemble des opérateurs
- S : coalition
- G : ensemble des gardiens actifs
- q_i : capacité effective de l’équipement de l’opérateur i
- F_i : coût fixe de fonctionnement en état actif
- beta_i : surcoût variable à pleine charge
- gamma_i : coût variable unitaire
- d(S) : demande de la coalition S
- t_i : volume de trafic acheminé par l’équipement de l’opérateur i
- G^*(S) : représentant optimal fixé parmi les ensembles de gardiens de S
- t^*(S) : répartition gloutonne associée à G^*(S)
- C^*(S) : coût minimal de la coalition
- C_i^*(N) : coût physique supporté par l’opérateur i dans la solution
  optimale retenue pour N
- C_i^0 : coût individuel de référence
- v(S) : économie réalisée par la coalition
- H : fenêtre finie d'heures
- d_i^h : demande de l'opérateur i à l'heure h
- G_H^*(S) : ensemble de gardiens optimal, fixe sur H
- C_H^*(S) : coût optimal de S sur H avec gardiens fixes
- \underline C_H^*(S) : borne obtenue en choisissant les gardiens à chaque heure
- v_H(S) : jeu des économies construit à partir de C_H^*
- z_i : part de la cagnotte attribuée à l’opérateur
- y_i : coût net final supporté par l’opérateur
- tau_i : transfert net total reçu par l’opérateur
- pour A inclus dans N, R^A désigne les vecteurs dont les coordonnées sont
  indexées par A
- les vecteurs physiques t sont globaux dans R^N, avec des zéros hors de
  la coalition considérée
- l’analyse de stabilité porte sur le jeu (N,v) ; les vecteurs z, y et tau
  appartiennent à R^N
- une allocation des économies est notée z et l’ensemble des allocations
  efficaces et individuellement rationnelles est noté A(N,v)

# Structure scientifique

- Partie 3 : modèle du système et ensembles admissibles
- Partie 4 : optimisation opérationnelle d’une coalition
- Partie 5 : analyse en théorie des jeux coopératifs
- Partie 6 : résultats numériques, encore à développer

# Résultats centraux

- Pour un ensemble de gardiens fixé, la répartition optimale remplit les
  capacités par ordre croissant des coûts variables unitaires.
- La sélection globale des gardiens est un problème à coûts fixes.
- Pour les petites coalitions, l’énumération exacte est acceptable.
- La sous-additivité est stricte lorsque l’un des gardiens de coût variable
  minimal des deux solutions séparées peut acheminer toute la demande réunie.
- Le jeu d’économies est défini par :
  v(S) = somme des coûts individuels de référence - coût coopératif minimal.
- Le nucléole et la valeur de Shapley sont étudiés.
- La non-vacuité générale du cœur n’est pas encore démontrée.

# Points ouverts

- justification et interprétation physique de la charge ;
- usure induite par une allocation déterministe répétée ;
- équité temporelle entre gardiens ;
- conditions de non-vacuité du cœur ;
- choix définitif des bornes de la fenêtre temporelle ;
- validation puis exécution du protocole numérique gelé ;
- cohérence entre optimisation opérationnelle et répartition des économies de coopération.
