---
title: "ジャックのレンタカー会社問題(強化学習第2版 4.3 方策反復)"
tags: ["Reinforcement Learning"]
date: 2023-01-10T15:00:00+09:00
---

# ジャックのレンタカー会社問題

ジャックのレンタカー会社問題について方策反復を用いて最適方策を求めた。本と同じ結果を得ることができた。

実装は、[reinforcement\_learning/main.cpp at main - niuez/reinforcement\_learning](https://github.com/niuez/reinforcement_learning/blob/fc57a8c63cdf4b6cb5c4632cf61cd0c89867e45a/jacks_rental_car/main.cpp)に載せてある。

# 練習問題4.7

上の拡張を行えば良い。$V(s)$についてプロットすると以下のようになった。

![](notes/reinforcement_img/ext_value_surface.png)

各状態における行動は以下のようになった。右向きに1つ目の支店の台数、下向きに2つ目の支店の台数、行動は「1つ目の支店の台数の変化」で表している。

```
 0  0  0  0  0  0  0  0  1  1  2  2  2  3  4  5  4  4  4  4  4 
 0  0  0  0  0  0  0  0  0  1  1  1  2  3  4  5  3  3  3  3  4 
 0  0  0  0  0  0  0  0  0  0  0  1  2  3  4  2  2  2  2  3  3 
 0  0  0  0  0  0  0  0  0  0  0  1  2  3  1  1  1  1  2  2  2 
 0  0  0  0  0  0  0  0  0  0  0  1  2  0  0  0  0  1  1  1  1 
 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-2  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-3 -3 -2  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-4 -3 -3  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-4 -4 -3 -3  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -4 -4 -3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 
-5 -5 -4 -4 -3 -3 -2 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 
-5 -5 -5 -4 -4 -3 -2 -2 -2  0  0 -2 -2 -2 -2 -2 -2 -2 -2  0  0 
-5 -5 -5 -5 -4 -3 -3 -3  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -4 -4 -4  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -5  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4 -3  0  0  0  0  1  0  0  0  0  0  0  0  0  0 
-5 -5 -5 -5 -5 -4 -3  0  0  0  0  0  0  0  0  0  0  0  0  0  0 
```

# 高速化について

この実装だと、$\sum_{s', r} p(s', r | s, a) (r + \gamma V(s'))$の計算に、車の最大数を$N$として$O(N^4)$かかる。

$$
\begin{aligned}
& \sum_{s', r} p(s', r | s, a) \lbrack r + \gamma V(s') \rbrack \\
=& \sum_{d_1 \leq i, s_1, d_2 \leq j, s_2} p_{\mathrm{dem1}}(d_1) p_{\mathrm{sup1}}(s_1) p_{\mathrm{dem2}}(d_2) p_{\mathrm{sup2}}(s_2) \lbrack \mathrm{move}(a) + \mathrm{r1}(d_1) + \mathrm{r2}(d_2) + \gamma V(i - d_1 + s_1, j - d_2 + s_2) \rbrack
\end{aligned}
$$

ただし、

- $s = (i, j)$: 支店1に$i$台、支店2に$j$台あって夜を迎えた状態
- $p_{dem1}(d_1)$: 支店1での1日の車の要求台数が$d_1$になる確率
- $\mathrm{move}(a)$: 行動$a$によって動く車の台数に応じた報酬
- ${\mathrm{r1}(d_1)}$: 支店1での要求台数$d_1$に応じた報酬

である。これを展開すると、

$$
\begin{aligned}
=& \mathrm{move}(a) + \sum_{d_1 \leq i} p_{\mathrm{dem1}}(d_1) \mathrm{r1}(d_1) + \sum_{d_2 \leq i} p_{\mathrm{dem2}}(d_2) \mathrm{r2}(d_2) + \sum_{x, y} \gamma Q_1(i, x) Q_2(j, y) V(x, y)
\end{aligned}
$$

ただし
$$
\begin{aligned}
Q_1(i, x) &= \sum_{x = s - d, d \leq i} p_{\mathrm{dem1}}(d) p_{\mathrm{sup1}}(s) \\
Q_2(j, y) &= \sum_{y = s - d, d \leq j} p_{\mathrm{dem2}}(d) p_{\mathrm{sup2}}(s)
\end{aligned}
$$

である。第2項、第3項、$Q_1, Q_2$によって畳み込みの係数を前計算しておけば、$O(N^2)$で処理することができる。

[高速化したコード](https://github.com/niuez/reinforcement_learning/blob/87c043c7beb7a7f78d0d71f43ac956d3de865592/jacks_rental_car/main.cpp)
