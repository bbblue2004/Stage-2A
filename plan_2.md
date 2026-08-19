# Annexe de travail — constats, historique, commentaires

Ce fichier recueille tout ce qui n'est pas le plan lui-même. `plan.md` ne
contient que le plan. Comme dans `plan.md`, les formules sont en texte brut
pour rester lisibles dans l'aperçu Markdown.

---

## 1. Statut du rapport

Les résultats numériques présents dans les Sections 6.3 à 6.5 proviennent du
protocole précédent, celui des 1 000 sites formés de quatre antennes réelles
appariées. Ils sont **obsolètes** et ne doivent pas être cités. Aucun résultat
de partage ou de stabilité ne sera écrit tant que les campagnes n'auront pas
été relancées.

---

## 2. Constats empiriques préalables

Vérifications faites sur `data/raw/radio_sites.csv`.

### 2.1 Étendue des données

| Élément | Valeur |
|---|---|
| Période | lundi 20 → dimanche 26 mars 2023 |
| Jours | 7 (5 ouvrés, 2 de week-end) |
| Identifiants radio | 3 825 |
| Lignes par jour | 91 800, soit 3 825 × 24 |
| Heures par antenne | 168 |

Les campagnes précédentes n'utilisaient que les 5 premiers jours, donc
uniquement des jours ouvrés, sans que ce soit explicité nulle part.

### 2.2 Nuage puissance–trafic

Scripts : `src/cli/weekday_weekend_scatter.py`,
`src/cli/intercept_support_check.py`.
Figures : `figures/diagnostics/weekday_weekend_scatter*.png`.

| Grandeur | Valeur |
|---|---|
| Heures à puissance nulle, exclues | 37 094 sur 642 600, soit 5,8 % |
| Heures actives | 605 506 |
| Heures actives à trafic exactement nul | 6 788, soit 1,12 % |
| Rapport trafic actif minimal sur pic, médiane | 0,019 |
| Antennes dont ce rapport dépasse 0,10 | 5,3 % |
| Antennes dont le week-end abaisse le minimum | 19,4 % |

Constats retenus, sans les développer dans le rapport :

- il n'y a pas d'étalement vertical à trafic nul, mais un palier horizontal au
  niveau de la puissance fixe, conforme au modèle affine ;
- la puissance fixe n'est donc pas extrapolée loin du support observé ;
- semaine et week-end suivent la même droite, ce qui justifie une calibration
  unique sur 7 jours.

---

## 3. Points écartés en cours de discussion

- **Campagne B** de stress des seuils : supprimée. Le balayage du seuil `k_h`
  est obtenu naturellement par la journée, la demande de pointe valant environ
  `r` fois la somme des capacités.
- **Validation hors échantillon** sur le week-end : abandonnée au profit d'un
  ajustement unique sur 7 jours.
- **Courbe de complexité** en fonction du nombre d'opérateurs : abandonnée.
  Seuls des temps pratiques pour `n = 3` et `n = 4` seront reportés.
- **Étude du biais d'activation partielle** : sans objet, puisque l'étalement
  vertical n'existe pas dans les données.
- **Condition locale de sous-additivité stricte** : conservée en simple
  remarque, non développée.

---

## 4. Journal des révisions

- **19 août 2026** — Création du plan. Suppression de la campagne B. Passage à
  7 jours. Grille `r` dans `{0,70 ; 0,80 ; 0,90 ; 1,00}` rétablie. Ajout du cas
  à trois opérateurs. Extension à 24 heures via l'additivité temporelle.
  Cascade de diagnostic réordonnée autour du test `E_Sh,H(N) <= 0`.
- **19 août 2026, rév. 2** — Intégration du week-end confirmée après contrôle
  visuel des nuages puissance–trafic.
- **19 août 2026, rév. 3** — Séparation en `plan.md`, qui ne contient que le
  plan, et `plan_2.md`, qui reçoit les constats et l'historique.
- **19 août 2026, rév. 4** — Toutes les formules passées en texte brut, sans
  LaTeX, pour l'aperçu Markdown.
