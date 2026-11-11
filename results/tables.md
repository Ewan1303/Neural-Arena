### leaderboard

| Task | PPO | Genetic algorithm | NEAT | Random actions |
|---|---|---|---|---|
| **Lander** (landing rate) | 100% · resmlp 4x64 tanh · 27 s | 99% · gru 64 · 52 s | 100% · sigmoid · 37 s | 0% |
| **Memory Lander** (landing rate) | 42% · gru 64 · 102 s | 76% · mlp 2x64 tanh · 30 s | 55% · relu · 39 s | 0% |
| **Drone** (waypoints reached) | 6.90 · mlp 4x64 tanh · 66 s | 2.50 · mlp 2x64 sin · 68 s | 1.94 · relu · 57 s | 0.10 |
| **Racer** (laps completed) | 2.51 · mlp 2x64 softsign · 19 s | 2.51 · mlp 2x64 relu · 33 s | 2.54 · relu · 49 s | 0.04 |

### cost of the best configuration

| Task | Method | Network | Score | Training time | Environment steps | Parameters |
|---|---|---|---|---|---|---|
| Lander | PPO | resmlp 4x64 tanh | 100% | 27 s | 0.4M | 13,188 |
| Lander | GA | gru 64 | 99% | 52 s | 4.4M | 14,084 |
| Lander | NEAT | sigmoid | 100% | 37 s | 7.5M | 60 |
| Memory Lander | PPO | gru 64 | 42% | 102 s | 0.6M | 14,468 |
| Memory Lander | GA | mlp 2x64 tanh | 76% | 30 s | 6.2M | 4,996 |
| Memory Lander | NEAT | relu | 55% | 39 s | 8.6M | 75 |
| Drone | PPO | mlp 4x64 tanh | 6.90 | 66 s | 1.0M | 13,508 |
| Drone | GA | mlp 2x64 sin | 2.50 | 68 s | 21.1M | 5,188 |
| Drone | NEAT | relu | 1.94 | 57 s | 17.3M | 95 |
| Racer | PPO | mlp 2x64 softsign | 2.51 | 19 s | 0.3M | 5,254 |
| Racer | GA | mlp 2x64 relu | 2.51 | 33 s | 4.1M | 5,254 |
| Racer | NEAT | relu | 2.54 | 49 s | 7.5M | 96 |

### activations (PPO, 2x64)

| Network | Lander | Drone | Racer | Overall (0 to 1) |
|---|---|---|---|---|
| leaky_relu | 100% | 6.37 | 2.47 | 0.97 |
| relu | 100% | 6.25 | 2.45 | 0.96 |
| softsign | 100% | 5.64 | 2.51 | 0.94 |
| tanh | 98% | 5.25 | 2.46 | 0.91 |
| sin | 100% | 4.97 | 2.44 | 0.90 |
| elu | 100% | 4.51 | 2.47 | 0.89 |
| gelu | 82% | 5.09 | 2.46 | 0.89 |
| gaussian | 13% | 5.71 | 2.33 | 0.84 |
| silu | 25% | 4.46 | 2.42 | 0.79 |
| identity | 34% | 2.13 | 2.46 | 0.71 |
| softplus | 0% | 3.21 | 1.48 | 0.61 |
| sigmoid | 0% | 1.92 | 0.36 | 0.28 |

### architectures (PPO, tanh)

| Network | Lander | Memory Lander | Drone | Racer | Overall (0 to 1) | Parameters (drone) | Training time (drone) |
|---|---|---|---|---|---|---|---|
| mlp 4x64 tanh | 100% | 29% | 6.90 | 2.39 | 0.96 | 13,508 | 66 s |
| resmlp 4x64 tanh | 100% | 34% | 6.33 | 2.44 | 0.95 | 13,508 | 65 s |
| gru 64 | 98% | 42% | 6.17 | 2.43 | 0.94 | 15,044 | 173 s |
| rnn 64 tanh | 94% | 36% | 6.19 | 2.37 | 0.93 | 5,124 | 80 s |
| mlp 2x64 tanh | 98% | 32% | 5.25 | 2.46 | 0.91 | 5,188 | 44 s |
| mlp 256 tanh | 42% | 21% | 2.02 | 2.48 | 0.75 | 4,100 | 50 s |
| mlp 64 tanh | 6% | 26% | 3.45 | 2.41 | 0.75 | 1,028 | 36 s |
| mlp 16 tanh | 0% | 0% | 3.24 | 2.41 | 0.64 | 260 | 36 s |

### mixed activations (PPO)

| Network | Lander | Drone | Racer | Overall (0 to 1) |
|---|---|---|---|---|
| 2x64 relu>tanh | 100% | 6.29 | 2.47 | 0.97 |
| 2x64 relu | 100% | 6.25 | 2.45 | 0.96 |
| 3x64 relu>sigmoid>tanh | 100% | 5.79 | 2.41 | 0.94 |
| 2x64 tanh | 98% | 5.25 | 2.46 | 0.91 |
| 2x64 sin>tanh | 74% | 5.47 | 2.47 | 0.91 |
| 2x64 tanh>relu | 33% | 6.17 | 2.49 | 0.88 |
| 2x64 gaussian>relu | 0% | 4.83 | 2.31 | 0.79 |
| 2x64 relu>softmax | 0% | 4.66 | 0.06 | 0.34 |
| 2x64 sigmoid>softmax | 0% | 0.27 | 0.00 | 0.08 |

### same network, PPO / GA

| Network | Lander | Memory Lander | Drone | Racer |
|---|---|---|---|---|
| mlp 2x64 sin | 100% / 100% | – / 77% | 4.97 / 2.50 | 2.44 / 2.12 |
| mlp 2x64 tanh | 98% / 100% | 32% / 76% | 5.25 / 1.77 | 2.46 / 2.37 |
| mlp 16 tanh | 0% / 99% | 0% / 63% | 3.24 / 1.89 | 2.41 / 2.46 |
| rnn 64 tanh | 94% / 99% | 36% / 27% | 6.19 / 2.09 | 2.37 / 2.02 |
| gru 64 | 98% / 99% | 42% / 32% | 6.17 / 2.10 | 2.43 / 2.15 |
| mlp 2x64 relu | 100% / 66% | – / 7% | 6.25 / 0.71 | 2.45 / 2.51 |
| mlp 2x64 sigmoid | 0% / 0% | – / 0% | 1.92 / 0.43 | 0.36 / 1.00 |
| mlp 2x64 sigmoid>softmax | 0% / 0% | – / 0% | 0.27 / 0.38 | 0.00 / 0.78 |

### NEAT variants

| Hidden-node activation | Lander | Memory Lander | Drone | Racer | Hidden nodes in champion | Connections in champion |
|---|---|---|---|---|---|---|
| tanh | 100% | 52% | 0.90 | 2.50 | 8.2 | 59 |
| relu | 98% | 55% | 1.94 | 2.54 | 10.8 | 66 |
| sigmoid | 100% | 49% | 1.60 | 2.52 | 8.1 | 62 |
| evolving | 100% | 48% | 1.37 | 2.51 | 7.6 | 56 |
| tanh recurrent | – | 52% | – | – | 8.7 | 47 |
| evolving recurrent | – | 38% | – | – | 7.7 | 52 |

### activations kept by evolving NEAT

| Activation | Hidden nodes | Share |
|---|---|---|
| sin | 24 | 21% |
| tanh | 20 | 18% |
| sigmoid | 18 | 16% |
| identity | 18 | 16% |
| relu | 15 | 13% |
| abs | 11 | 10% |
| gaussian | 8 | 7% |

total 114

### totals

runs: 408
configurations: 136
total training CPU time: 4.09 hours
total environment steps: 1.37 billion

ppo: 258 runs, 2.33 h, 146M steps
ga: 96 runs, 0.99 h, 684M steps
neat: 54 runs, 0.77 h, 536M steps