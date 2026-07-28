# Implemented model

For each operator `i`, the code uses:

- demand `d_i`;
- effective capacity `q_i`;
- avoidable fixed active cost `F_i`;
- unit variable cost `gamma_i`;

For a non-empty coalition `S`, a feasible guardian set `G` has enough
aggregate capacity to carry `d(S)`. At fixed `G`, the greedy allocation fills
guardians in increasing `gamma_i` order. The selected solution minimizes

```text
C(S,G,t) = sum_{i in G} (F_i + gamma_i t_i).
```

Enumeration of all feasible guardian sets gives `C*(S)`. The empty-set
convention is `C*(empty) = 0`.

The standalone reference is

```text
C_i^0 = F_i + gamma_i d_i = C*({i}).
```

The transferable-utility game is the operational savings game

```text
v(S) = sum_{i in S} C_i^0 - C*(S),   v(empty) = 0.
```

For the grand coalition, the program computes one vector `z` in the core:

```text
sum_i z_i = v(N)
sum_{i in Q} z_i >= v(Q)  for every non-empty proper Q
z_i >= 0.
```

It independently solves the balanced-family linear program

```text
B(N) = max sum_Q lambda_Q v(Q)
```

subject to non-negative weights and unit total coverage of every player. The
Bondareva--Shapley theorem gives a non-empty core exactly when `B(N)=v(N)`.

Finally,

```text
y_i   = C_i^0 - z_i
tau_i = C_i^*(N) - y_i.
```

A positive `tau_i` is received and a negative `tau_i` is paid. Efficiency of
`z` implies `sum_i tau_i = 0`.

The least-core implementation is outside the current scope and is not executed.
