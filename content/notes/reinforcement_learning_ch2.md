---
title: "k本腕バンディット問題(強化学習第2版 2章)"
tags: ["Reinforcement Learning"]
date: 2023-01-09T15:00:00+09:00
---

# 強化学習第2版を買った

今まで焼きなまし法や遺伝的プログラミングによる、「最終的な結果に対する評価」によって最適化を行う手法を勉強してきたが、「一手一手に対する評価」はどのようにするのか興味が湧いたので勉強してみる。

# k本腕バンディット問題に対するアプローチ

ε-greedy法、楽観的初期値をもつε-greedy法、上限信頼区間(UCB)行動選択法、確率的勾配上昇法を実装して、得られた報酬の平均と最適行動の割合を各ステップについて計算しグラフにした。

![](./notes/reinforcement_img/k_armed_bandit.png)

実装は、[reinforcement\_learning/main.cpp at main - niuez/reinforcement\_learning](https://github.com/niuez/reinforcement_learning/blob/8240516bdfd47227fb44fd4f2f6524f6b1ab06b3/k-armed-bandit/main.cpp)に載せてある。


