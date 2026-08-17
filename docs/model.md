# Implemented model

For each operator `i` and hour `h`, the code uses demand `d_i^h`, effective
capacity `q_i`, avoidable fixed active cost `F_i`, and unit variable cost
`gamma_i`.

For a non-empty coalition `S`, a feasible hourly guardian set `G_h` has enough
aggregate capacity to carry `d^h(S)`. At fixed `G_h`, the greedy allocation
fills guardians in increasing `gamma_i` order and minimizes

```text
C_h(S,G_h,t^h) = sum_{i in G_h} (F_i + gamma_i t_i^h).
```

Enumeration of all feasible guardian sets independently at each hour gives

```text
C_H*(S) = sum_{h in H} C_h*(S),   C_H*(empty) = 0.
```

The code also computes the secondary persistent policy `C_H,pers*(S)`, where
one guardian set must be used at every hour. It always satisfies
`C_H*(S) <= C_H,pers*(S)` and does not define the main game.

The standalone window reference is

```text
C_H*({i}) = sum_h (F_i + gamma_i d_i^h).
```

The transferable-utility game is the hourly operational savings game

```text
v_H(S) = sum_{i in S} C_H*({i}) - C_H*(S),   v_H(empty) = 0.
```

For the grand coalition, a core allocation `z` satisfies

```text
sum_i z_i = v_H(N)
sum_{i in Q} z_i >= v_H(Q)  for every non-empty proper Q
z_i >= 0.
```

The implementation independently solves the balanced-family linear program.
The Bondareva--Shapley theorem gives a non-empty core exactly when its optimum
equals `v_H(N)`. It also computes convexity, Shapley, the least core and the
nucleolus. By default it selects Shapley when Shapley belongs to the core, and
the nucleolus otherwise.

For the chosen hourly optimum, physical costs are summed over the hourly
guardian sets. Net costs and transfers are

```text
y_i   = C_H*({i}) - z_i
tau_i = C_i,H*(N) - y_i.
```

A positive `tau_i` is received and a negative `tau_i` is paid. Efficiency of
`z` implies `sum_i tau_i = 0`.
