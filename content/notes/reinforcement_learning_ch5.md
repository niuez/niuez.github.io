---
title: "ブラックジャック問題(強化学習第2版 5 方策オン/オフ型)"
tags: ["Reinforcement Learning"]
date: 2023-01-11T15:00:00+09:00
---

# ブラックジャック問題

方策オン型初回訪問MCと、方策オフ型重み付き重点サンプリングMCを実装した。コードは[reinforcement\_learning/main.cpp at main · niuez/reinforcement\_learning](https://github.com/niuez/reinforcement_learning/blob/99049126c8144ccd1581987e827b69dae4977d65/blackjack/main.cpp)に載せてある。両方、ソフト方策はεソフトを用いている。収束は方策オフ型の方がはやかった。

両者の違いは後者が、グリーディーな方策と、εによって選ばれたランダムな方策による価値の更新の重みが異なることだ。前者は同一に扱っている。

- 1: ヒット(もう一枚)
- 0: スティック(やめ)

```
ace = 0
     A23456789T
11 : 1111111111
12 : 1110001111
13 : 1000001111
14 : 1000001111
15 : 1000001111
16 : 1000001111
17 : 0000000000
18 : 0000000000
19 : 0000000000
20 : 0000000000
21 : 0000000000
ace = 1
     A23456789T
11 : 0000000000
12 : 1111111111
13 : 1111111111
14 : 1111111111
15 : 1111111111
16 : 1111111111
17 : 1111111111
18 : 1000000011
19 : 0000000000
20 : 0000000000
21 : 0000000000
```

価値関数について、ヒットするかスティックするかの差分を計算してみた。赤ほどヒットした方がよく、青ほどスティックした方がいい。


### 使用可能なエースなし

![](./notes/reinforcement_img/no_ace_value.png)

### 使用可能なエースあり

![](./notes/reinforcement_img/ace_value.png)