# Journal de recherche

## 26 juillet 2026

- Installation de VS Code, MiKTeX, LaTeX Workshop et Codex terminée.
- Le projet compile localement.
- Correction de \alphi en \alpha.
- Les commandes IEEEauthorblock ont été adaptées à la classe article.
- Plusieurs références et citations restent à corriger.

## Décisions actives

- Le modèle horairement reconfigurable est le modèle principal ; l'engagement
  persistant des gardiens est une extension opérationnelle secondaire.
- L’usure, les coûts de commutation et la rotation des gardiens restent des
  limites discutées explicitement.
- Ne plus modifier les notations fondamentales sans nécessité scientifique.

## 4 août 2026 — Jour 1

- IEEE TGCN retenue comme cible principale ; le passage au template IEEE est
  reporté à la phase de rédaction finale.
- Message central gelé : économies énergétiques et stabilité coalitionnelle
  doivent être étudiées conjointement.
- Quatre contributions et cinq questions de recherche initialement formulées.
- Nucléole initialement retenu comme règle principale orientée stabilité ;
  choix révisé le 5 août.
- Étude qualifiée de semi-empirique ; provenance détaillée des données laissée
  à compléter.
- Abandon du profil principal obtenu par trois copies faiblement bruitées : le
  scénario principal tirera trois profils Orange distincts autour d’une
  antenne d’ancrage, sur vingt graines.
- Capacité traitée comme paramètre de sensibilité au moyen du taux de charge
  au pic, et non comme valeur physique estimée par la constante 0,75.
- Une sensibilité à la consommation de veille avait initialement été prévue ;
  elle est abandonnée dans la révision du 5 août.
- Les bornes de la fenêtre restent paramétrables et seront choisies plus tard
  à partir du trafic, avant l’étude de stabilité.
- Politique principale : un même ensemble de gardiens pendant toute la
  fenêtre, avec allocation du trafic variable par heure. L’optimum horaire
  indépendant devient une borne basse.
- Baselines gelées : aucun partage, R-to-1, heuristique de capacité, optimum
  persistant exact ; l’optimum horaire est une borne idéale.
- L’ordre de priorité mathématique est confirmé : conserver le cas homogène,
  conserver le certificat leave-one-out dans le sens écrit dans le bilan, et
  ajouter au jour 2 un contre-exemple montrant que le très faible trafic ne
  garantit pas l’appartenance de Shapley au cœur en présence d’hétérogénéité.
- Le code sélectionnait alors le nucléole par défaut ; cette décision est
  révisée le 5 août.
- Protocole complet consigné dans `NUMERICAL_PROTOCOL.md`.

## 5 août 2026 — Révision du jour 1

- Puissance de veille fixée à \(0\ \mathrm{W}\) ; suppression de la sensibilité
  au paramètre de veille.
- Règle de partage principale révisée : Shapley lorsqu'elle appartient au
  cœur, nucléole sinon. Le nucléole reste stable lorsque le cœur est non vide ;
  si le cœur est vide, aucune allocation parfaitement stable n'existe.
- La section numérique adopte la structure de Bousia et Koutitas : scénario,
  validation, économies, sensibilité, puis stabilité et partage. Les questions
  de recherche explicites et les identifiants d'implémentation sont retirés de
  la rédaction scientifique.
- L'extension temporelle est déplacée dans les Sections 3--5 : définition de
  \(H\) et des gardiens fixes en Section 3, coût \(C_H^*\) en Section 4 et jeu
  \(v_H\) en Section 5. La Section 6 ne redéfinit plus ces objets.
- L'expression « coût persistant » est abandonnée dans le texte : \(C_H^*\)
  est simplement le coût optimal sur la fenêtre, sous la contrainte de
  conserver les mêmes gardiens.
- La projection normalisée de Shapley est définie en Section 5 et n'est pas
  qualifiée de \emph{fairest core}, faute de propriété axiomatique justifiant
  ce terme.
- Relecture de fin de Day 1 : ajout d'une proposition formelle de
  sous-additivité de \(C_H^*\), extension au jeu \(v_H\) de la condition
  suffisante de capacité, et définition explicite de la face du moindre cœur
  utilisée pour projeter Shapley.
- La Section 6 comprend désormais un test préalable de cohérence entre
  algorithmes et formulations théoriques, cinq tests numériques, une baseline
  d'allocation proportionnelle et une discussion des limites.

## 17 août 2026 — Inversion du modèle temporel principal

- Le coût principal devient \(C_H^*(S)=\sum_h C_h^*(S)\), avec sélection des
  gardiens et allocation du trafic indépendantes à chaque heure.
- Le coût \(C_{H,\mathrm{pers}}^*(S)\) est conservé comme extension secondaire
  et son écart à \(C_H^*(S)\) mesure le prix de la persistance.
- Les preuves de NP-difficulté, sous-additivité, superadditivité et les deux
  contre-exemples sont conservées. La réduction semi-homogène utilise la somme
  des plafonds horaires ; la preuve de cœur non vide en faible trafic est
  reformulée heure par heure.
- Toute la campagne a été reconstruite : 20 000 instances opérationnelles,
  20 000 jeux et 182 000 lignes de sensibilité.
- L'économie horaire médiane vaut 70,6 %. Aucun cœur vide n'est observé dans
  la campagne principale ; Shapley est hors du cœur dans 3,405 % des cas.
  Des cœurs vides subsistent dans certaines sensibilités à la taille de la
  coalition et à la fenêtre.
