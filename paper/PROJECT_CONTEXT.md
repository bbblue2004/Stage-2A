# Objet de l’article

L’article étudie le partage opportuniste d’infrastructures RAN entre
opérateurs mobiles, à l’aide de l’optimisation combinatoire et de la
théorie des jeux coopératifs.

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
- C^*(S) : coût minimal de la coalition
- C_i^0 : coût individuel de référence
- w(S) : économie réalisée par la coalition

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
- Le jeu d’économies est défini par :
  w(S) = somme des coûts individuels de référence - coût coopératif minimal.
- Le nucléole et la valeur de Shapley sont étudiés.
- La non-vacuité générale du cœur n’est pas encore démontrée.

# Points ouverts

- justification et interprétation physique de la charge ;
- usure induite par une allocation déterministe répétée ;
- équité temporelle entre gardiens ;
- conditions de non-vacuité du cœur ;
- protocole des résultats numériques ;
- cohérence entre optimisation opérationnelle et répartition des économies de coopération.
