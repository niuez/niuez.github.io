---
title: "風が吹くグリッドワールド(強化学習第2版 6 Sarsa/Q-learning/expected Sarsa)"
tags: ["Reinforcement Learning"]
date: 2023-01-15T15:00:00+09:00
---

# 風が吹くグリッドワールド

$10 \times 7$のグリッド上で$x = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9$を満たすグリッドでは、それぞれ$0, 0, 0, 1, 1, 1, 2, 2, 1, 0$の強さの風が$y$のマイナスの向きに吹いている。風が吹いているマスでは、そこから移動しようとした時に風の分が加わった異動になってしまう。この世界で、$(0, 3)$から$(7, 3)$に到達する必要があるときの最善手を求めたい。移動は、4方向に限ることにした。

Sarsa法(方策オンTD(0))、Q-learning(方策オフTD(0))、期待Sarsa法を実装して比較を行った。コードは[reinforcement\_learning/main.cpp at main · niuez/reinforcement\_learning](https://github.com/niuez/reinforcement_learning/blob/ab99c2c04d9a66049df6687923a5ddad419d77bd/grid_wind_world/main.cpp)に載せてある。

それぞれの計算方法で得られた方策の価値をベクトルにして表示させてみた。どの計算結果でも、一度右に進み続け風のない右端で上に移動しゴールを狙っている。

### Sarsa法

![](./notes/reinforcement_img/action_quiver.png)

### Q-learning

![](./notes/reinforcement_img/action_quiver_Q.png)

### 期待Sarsa法

![](./notes/reinforcement_img/action_quiver_exp.png)


