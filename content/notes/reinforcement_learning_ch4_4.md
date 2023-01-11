---
title: "ギャンブラー問題(強化学習第2版 4.4 価値反復)"
tags: ["Reinforcement Learning"]
date: 2023-01-11T15:00:00+09:00
---

# ギャンブラー問題

$p_h=0.4,0.25,0.55$それぞれのギャンブラー問題について、価値反復を用いて最適方策を求めた。

実装は、[reinforcement\_learning/main.cpp at main - niuez/reinforcement\_learning](https://github.com/niuez/reinforcement_learning/blob/836f80e23753f79988ca75515ce45fdbae6d1f84/gambler/main.cpp)に載せてある。

価値関数と、最適方策を図にした。

## $p_h=0.4$

![](notes/reinforcement_img/p0.400000.png)

## $p_h=0.25$

![](notes/reinforcement_img/p0.250000.png)

$p_h \leq 0.5$では、どこかで賭けをして勝ちを狙いに行く必要があるっぽい。

## $p_h=0.55$

![](notes/reinforcement_img/p0.550000.png)

$p_h=0.55$については、少しずつ掛けて勝つことができるのでこのような結果になった。
