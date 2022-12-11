---
title: "Genetic Local Search + MSXFの勉強とIntroduction to Heuristic Contest"
tags: ["Genetic Algorithm"]
date: 2022-12-11T15:00:00+09:00
---

# 概要

遺伝的アルゴリズム(Genetic Algorithm)の練習として[Introduction to Heuristic Contestの問題](https://atcoder.jp/contests/intro-heuristics/tasks/intro_heuristics_a)を解いてみました。最終的なスコアは117Mで焼きなましの124Mには劣りますが、良い成績は得られていそうです。

[前回]({{< ref "/notes/genetic_local_search_tsp" >}})のも参考に

# 方針

[遺伝的局所探索法によるジョブショップスケジューリング問題の解法](https://cir.nii.ac.jp/crid/1050001337886598400)で述べられている

- Genetic Local Search(遺伝的局所探索, GLS)
- Multi Step Crossover Fusion(多段階探索交叉, MSXF)

の2つの手法を取り入れました。

# Genetic Local Search

各世代を作り終わった後に、局所探索を行うことで個体を改善することを試みます。{{ref . "genetic_local_search_tsp.md" }}も参考にしてください。

# Multi Step Crossover Fusion

交叉するときによりよい交叉を探索によって求め子孫を作る、というのがMSXFの基本的なお気持ちです。親二つを交叉してできる子孫の空間を探索する際、親xからの距離によって子孫が「どれくらい親xっぽいか」を定義し、探索を進めるにつれて「どれくらい親xに似せるか」を絞っていきます。空間を探索している際、その遷移を受理するかしないかは、焼きなましと同じ方法で行います。

# 解法の交叉の方法

交叉は1点交叉を用いました(2交叉も試しましたが、微妙に1点交叉の方が強かったです)。論文のMSXFをそのまま採用すると実行時間が足りないので、次のようにアレンジしました。

1. 親$x$の前半分、親$y$の後ろ半分を1点交叉したものを初期解$c$とする。$n = D / 2$とする。
2. $m = n + \mathrm{rand}(-15, 15)$として、$m$で交叉したものを$p$とする。
3. 確率$e^{(\mathrm{score}(p) - \mathrm{score}{c}) / T}$で受理
4. 1-3を指定回数繰り返す

これを採用すると113Mから117Mまで伸びます。

[提出](https://atcoder.jp/contests/intro-heuristics/submissions/37220631)

{{< details 提出コード >}}
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <random>
#include <iomanip>
#include <bits/stdc++.h>

struct Timer {
  std::chrono::high_resolution_clock::time_point st;
  Timer() { st = now(); }
  std::chrono::high_resolution_clock::time_point now() { return std::chrono::high_resolution_clock::now(); }
  std::chrono::milliseconds::rep span() {
    auto ed = now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
  }
};

using u32 = std::uint_fast32_t;
using i64 = std::int_fast64_t;
constexpr int K = 26;
int D;
std::vector<i64> C;
std::vector<std::vector<i64>> S;

using Score = i64;

struct Individual {
  std::vector<int> vec;

  Individual() = default;
  explicit Individual(std::vector<int> vec): vec(std::move(vec)) {}

  Individual crossover(std::mt19937& mt, const Individual& y) const {
    Individual mx;
    Score score = -1e9;
    int now = vec.size() / 2;
    {
      std::vector<int> next(vec.size());
      std::copy(vec.begin(), vec.begin() + vec.size() / 2, next.begin());
      std::copy(y.vec.begin() + vec.size() / 2, y.vec.end(), next.begin() + vec.size() / 2);
      Individual child(std::move(next));
      score = child.evaluate_state();
      mx = std::move(child);
    }

    constexpr int Count = 30;
    for(int q = 0; q < Count; q++) {
      int DIFF = 15;
      int at = std::min(D - 1, std::max(1, std::uniform_int_distribution<>(-DIFF, DIFF - 1)(mt) + now));
      if(at >= 0) at++;
      std::vector<int> next(vec.size());
      std::copy(vec.begin(), vec.begin() + at, next.begin());
      std::copy(y.vec.begin() + at, y.vec.end(), next.begin() + at);
      Individual child(std::move(next));
      Score next_score = child.evaluate_state();
      if(score <= next_score || std::bernoulli_distribution(std::exp(-(score - next_score) / 4e2))(mt)) {
        score = next_score;
        mx = std::move(child);
        now = at;
      }
    }
    return mx;
  }

  void mutation(std::mt19937& mt) {
    std::bernoulli_distribution swap_dist(0.005);
    for(int i = 1; i < vec.size(); i++) {
      if(swap_dist(mt)) {
        std::uniform_int_distribution<> at_dist(std::max(0, i - 15), i - 1);
        int j = at_dist(mt);
        std::swap(vec[i], vec[j]);
      }
    }
    for(int i = 0; i < vec.size(); i++) {
      if(swap_dist(mt)) {
        vec[i] = std::uniform_int_distribution<>(0, K - 1)(mt);
      }
    }
  }

  Score evaluate_state() const {
    i64 sum = 0;
    std::vector<i64> before(K, -1);
    for(int i = 0; i < D; i++) {
      int v = vec[i];
      sum += S[i][v];
      i64 d = i - before[v];
      sum -= C[v] * d * (d - 1) / 2;
      before[v] = i;
    }
    for(int v = 0; v < K; v++) {
      i64 d = D - before[v];
      sum -= C[v] * d * (d - 1) / 2;
    }
    return sum;
  }
};



void local_search(std::mt19937& mt, Individual& x, Score& score) {
  constexpr int Count = 50;
  constexpr double T = 5e2;
  for(int q = 0; q < Count; q++) {
    if(std::bernoulli_distribution(0.5)(mt)) {
      int i = std::uniform_int_distribution<>(0, x.vec.size() - 1)(mt);
      int j = std::uniform_int_distribution<>(std::max(0, i - 13), std::min(D - 2, i + 12))(mt);
      if(j >= i) j++;
      std::swap(x.vec[i], x.vec[j]);
      Score next = x.evaluate_state();
      if(score <= next || std::bernoulli_distribution(std::exp(-(score - next) / T))(mt)) {
        score = next;
      }
      else {
        std::swap(x.vec[i], x.vec[j]);
      }
    }
    else {
      int i = std::uniform_int_distribution<>(0, x.vec.size() - 1)(mt);
      int before = x.vec[i];
      x.vec[i] = std::uniform_int_distribution<>(0, K - 2)(mt);
      if(before <= x.vec[i]) x.vec[i]++;
      Score next = x.evaluate_state();
      if(score <= next || std::bernoulli_distribution(std::exp(-(score - next) / T))(mt)) {
        score = next;
      }
      else {
        x.vec[i] = before;
      }
    }
  }
}

struct Generation {
  constexpr static int Count = 12;
  constexpr static int Elite = 4;
  constexpr static int NewInd = 3;
  std::vector<std::pair<Individual, Score>> inds;

  void init(std::mt19937& mt) {
    for(int i = 0; i < Count; i++) {
      std::vector<int> init(D);
      for(int d = 0; d < D; d++) {
        init[d] = std::uniform_int_distribution<>(0, K - 1)(mt);
      }
      Individual x(std::move(init));
      Score score = x.evaluate_state();
      inds.emplace_back(std::move(x), std::move(score));
    }
  }

  Generation next_gen(std::mt19937& mt) const {
    Generation next;

    std::vector<int> idx(Count);
    std::iota(idx.begin(), idx.end(), 0);

    // elitism
    std::nth_element(idx.begin(), idx.begin() + Elite - 1, idx.end(), [&](int i, int j) { return inds[i].second > inds[j].second; });
    for(int i = 0; i < Elite; i++) {
      next.inds.push_back(inds[idx[i]]);
    }

    // selection & crossover by roulette
    Score max_score = std::max_element(inds.begin(), inds.end(), [](auto& a, auto& b) { return a.second > b.second; })->second;
    std::vector<double> pie(inds.size());
    for(int i = 0; i < inds.size(); i++) {
      pie[i] = inds[i].second - max_score;
      if(i > 0) {
        pie[i] += pie[i - 1];
      }
    }
    std::uniform_real_distribution<> dice(0, pie.back());
    for(int i = Elite; i < Count - NewInd; i++) {
      int x = std::lower_bound(pie.begin(), pie.end(), dice(mt), [](auto& a, double v) { return a < v; }) - pie.begin();
      double diff = pie[x] - (x == 0 ? 0 : pie[x - 1]);
      int y = std::lower_bound(pie.begin(), pie.end(), std::uniform_real_distribution<>(0, pie.back() - diff)(mt),
          [&](auto& a, double v) { return (a >= pie[x] ? a - diff : a ) < v; }
        ) - pie.begin();
      Individual child = inds[x].first.crossover(mt, inds[y].first);
      child.mutation(mt);
      Score score = child.evaluate_state();
      next.inds.emplace_back(std::move(child), std::move(score));
    }
    for(int i = Count - NewInd; i < Count; i++) {
      std::vector<int> init(D);
      for(int d = 0; d < D; d++) {
        init[d] = std::uniform_int_distribution<>(0, K - 1)(mt);
      }
      Individual x(std::move(init));
      /*
      Individual x(next.inds[i - Count + NewInd].first);
      x.mutation(mt);
      x.mutation(mt);
      */
      Score score = x.evaluate_state();
      next.inds.emplace_back(std::move(x), std::move(score));
    }
    for(auto& [x, score]: next.inds) {
      local_search(mt, x, score);
    }
    return next;
  }
};

int main() {
  std::mt19937 mt;
  //const int Century = 100000;
  std::cin >> D;
  C.resize(K);
  for(int i = 0; i < K; i++) {
    std::cin >> C[i];
  }
  S.resize(D, std::vector<i64>(K));
  for(int i = 0; i < D; i++) {
    for(int j = 0; j < K; j++) {
      std::cin >> S[i][j];
    }
  }
  Generation gen;
  gen.init(mt);

  Timer timer;

  //for(int q = 0; q < Century; q++) {
  int q = 0;
  Individual MAX;
  Score score = -1e9;
  while(timer.span() < 1980) {
    gen = gen.next_gen(mt);
    q++;
    if(q % 100 == 0) {
      auto& NEXT = *std::max_element(gen.inds.begin(), gen.inds.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
      if(score < NEXT.second) {
        MAX = NEXT.first;
        score = NEXT.second;
      }
      std::cerr << q << " " << score << std::endl;
    }
    //std::cerr << std::fixed << std::setprecision(10) << q++ << "\t" << MAX.second << std::endl;
  }
  std::cerr << timer.span() << std::endl;
  std::cerr << score << std::endl;
  for(int i = 0; i < D; i++) {
    std::cout << MAX.vec[i] + 1 << "\n";
  }
}
```

{{</details>}}



