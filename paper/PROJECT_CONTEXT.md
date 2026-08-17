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
- Fenêtre numérique : sept créneaux nocturnes [00:00, 07:00) pour chacun des
  cinq jours ; les gardiens sont fixes pendant une nuit et peuvent changer la
  nuit suivante. Cette fenêtre ne varie pas entre les scénarios de la campagne
  principale, mais ses bornes restent des paramètres du code.
- Période empirique : les cinq premiers jours observés, soit 120 créneaux
  horaires par antenne, sont utilisés dans toutes les expériences.
- Population simulée : 1 000 sites virtuels de quatre antennes, générés une
  fois avec la graine 20260814 puis réutilisés dans tous les scénarios.
- Politique opérationnelle principale : même ensemble de gardiens sur toute
  la fenêtre, allocations de trafic horaires.
- Référence énergétique : l'absence de partage sert uniquement à normaliser
  l'énergie évitée. Les comparaisons opérationnelles portent sur l'activation
  par capacité décroissante et la répartition proportionnelle sur les gardiens optimaux.
  L'optimum principal fixe les gardiens sur la nuit et applique l'allocation
  gloutonne à chaque heure ; l'optimum horaire est seulement une borne basse.
- Capacité : paramètre de scénario construit de sorte que le pic observé
  représente entre 70 % et 100 % de \(q_i\) ; aucune capacité physique
  n’est prétendue observée. La demande n’est pas multipliée dans la campagne
  principale ; ses variations sont réservées à l’analyse de sensibilité.
- Sensibilité : une section globale, placée après les résultats opérationnels
  et de stabilité, varie séparément capacité, trafic, coefficients de
  puissance, veille, fenêtre et nombre d'opérateurs. Le prix commun de
  l'électricité est traité par invariance d'échelle.

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
- Sur les 20 000 instances de la Section 6.3, l'optimum persistant évite une
  médiane de 67,7 % de l'énergie autonome et retient un seul gardien dans
  64,5 % des cas.
- Sur ces mêmes 20 000 instances, le cœur est vide dans 25,2 % des cas ; il
  est non vide mais exclut Shapley dans 12,0 % des cas, et Shapley appartient
  au cœur dans 62,9 % des cas.
- L'analyse de sensibilité repose sur 1 000 sites (5 jours, et 4 nuits communes
  pour la position de la fenêtre) et sur 1 000 sites stratifiés pour chaque
  taille de coalition entre 2 et 6, soit 182 000 lignes de résultats.
- Sur les plages testées, la position et la durée de la fenêtre sont les
  principaux leviers opérationnels. Les perturbations uniformes de +/-20 % de
  F_i ou gamma_i changent l'économie médiane de moins de 3 points et la
  fréquence de stabilité de moins d'un point.
- Entre quatre et six opérateurs, l'économie médiane reste voisine de 69 %,
  tandis que la fréquence d'une valeur de Shapley dans le cœur passe de
  63,2 % à 32,7 %.
- La condition de capacité individuelle suffisante garantit bien un cœur non
  vide dans toutes les instances où elle s'applique, sans garantir Shapley ;
  le certificat leave-one-out détecte 71,5 % des cœurs vides sans faux positif.
- La sous-additivité est stricte lorsque l’un des gardiens de coût variable
  minimal des deux solutions séparées peut acheminer toute la demande réunie.
- Le coût \(C_H^*\) avec gardiens fixes est sous-additif et le jeu \(v_H\)
  est superadditif.
- Le jeu d’économies est défini par :
  v(S) = somme des coûts individuels de référence - coût coopératif minimal.
- Le nucléole et la valeur de Shapley sont étudiés.
- Si chaque opérateur peut servir la demande totale à chaque heure de \(H\),
  le cœur de \((N,v_H)\) est non vide.
- La non-vacuité générale du cœur n’est pas encore démontrée.

# Points ouverts

- justification et interprétation physique de la charge ;
- usure induite par une allocation déterministe répétée ;
- équité temporelle entre gardiens ;
- conditions de non-vacuité du cœur ;
- validation puis exécution du protocole numérique gelé ;
- cohérence entre optimisation opérationnelle et répartition des économies de coopération.
