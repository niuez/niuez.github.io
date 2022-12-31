---
title: "NSGA-IIによるMulti Knapsack Problemへのアプローチ"
tags: ["Genetic Algorithm"]
date: 2022-12-31T15:00:00+09:00
---

# 概要


[前回勉強したLexicase Selection入門の話]({{< ref "/notes/lexicase_booleanCSP" >}})では、多目的関数に対する最適化を行った。
今回は、同じく多目的関数に対する手法であるNon-dominated Sorting Genetic Algorithm II(NSGA-II)を勉強したのでメモしておく。

参考

1. [A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II](https://web.njit.edu/~horacio/Math451H/download/2002-6-2-DEB-NSGA-II.pdf)
2. [多数目的最適化問題における進化型多目的最適化 アルゴリズムの問題点とその改良手法に関する考察](https://www.jstage.jst.go.jp/article/iscie/22/6/22_6_220/_pdf)
3. [多目的最適化問題のNSGA-Ⅱを勉強したの整理しました - Qiita](https://qiita.com/DS27/items/025a52b26a9f2471e67c)
4. [多目的な最適化問題を解くためにNSGA-Ⅱを解説する - Qiita](https://qiita.com/Taichi-Furukawa/items/a7a0982cc20a401133c0)

# Non-dominated Sorting

日本語では非優劣ソートと呼ばれる。解の中でパレート最適な集合をrank0として、それを取り除き再帰的にrank1, ...を決めていくソートである。これを用いて多目的関数に対するElisitmを実現する。
参考1の論文では$O(M N^2)$での実装が示されていて、それを用いて以下では実装している。

# Crowding Distance

日本語では混雑度といい、各個体の周囲にある個体がどの程度密集しているかを評価するために用いる。計算方法は参考1で示されていて、以下である。

1. すべての個体の混雑度を0で初期化する
2. 1番目の評価基準$f$について以下を行う
3. $f$による評価で個体をソートする
4. ソートした時の両端の個体(最小値と最大値を持つ個体)の混雑度に$\infty$を加算する
5. それ以外のi番目の個体の混雑度には$\frac{f(i + 1) - f(i - 1)}{f^{\mathrm{max}} - f^{\mathrm{min}}}$を加算する
6. 2番目以降の評価基準についても2~5を行う

混雑度が大きいほど、周囲に同じような個体が存在していないことになる。

# NSGA-IIの流れ

NSGAはNon-dominated SortingとCrowding Distanceを用いて以下のように世代更新を行う。

- $P$: 現在の世代
- $Q$: 次の世代、最初は空
- $N=|P|$
- $E$: Elitismで残す個体数

1. $P$に対してNon-dominated Sorting、rank$i$の個体の集合を$F_i$とする
2. $i = 0$から$|R| + |F_i| \leq E$なら$R := R \cup $F_i$
3. $F_i$に対してCrowding Distanceを計算し、混雑度が高い個体から$|R| = E$になるまで$R$に追加する
4. $Q := R$
5. $R$から(Rank, Crowding Distance)のタプルによる順序によってトーナメント選択を行い、交叉、突然変異を行った個体を$Q$に追加する
6. $|R| + |Q| = N$になるまで追加する

# 実装

Multi Knapsack Problem(参考2)に対してNSGA-IIを適用した実装を以下に示す。このようなパレート最適な解が見つかった。

![](./notes/multikp.png)

{{< details 提出コード >}}
```cpp
#include <vector>
#include <iostream>
#include <numeric>
#include <random>

template<class T>
std::vector<std::vector<int>> non_dominated_sort(const std::vector<std::vector<T>>& p, int cnt) {
  const int N = p.size();
  const int M = p.empty() ? 0 : p.front().size();
  std::vector<int> n(N);
  std::vector<std::vector<int>> S(N);
  std::vector<std::vector<int>> F(1);
  for(int i = 0; i < N; i++) {
    for(int j = i + 1; j < N; j++) {
      int ic = 0;
      int id = 0;
      int jc = 0;
      int jd = 0;
      for(int k = 0; k < M; k++) {
        T x = p[i][k];
        T y = p[j][k];
        if(x <= y){
          ic++;
          if(x < y) {
            id++;
          }
        }
        if(x >= y){
          jc++;
          if(x > y) {
            jd++;
          }
        }
      }
      if(ic == M && id > 0) {
        S[i].push_back(j);
        n[j]++;
      }
      if(jc == M && jd > 0) {
        S[j].push_back(i);
        n[i]++;
      }
    }
    if(n[i] == 0) {
      cnt--;
      F[0].push_back(i);
    }
  }
  while(F.back().size() > 0 && cnt > 0) {
    std::vector<int> Q;
    for(auto i: F.back()) {
      for(auto j: S[i]) {
        n[j]--;
        if(n[j] == 0) {
          Q.push_back(j);
          cnt--;
        }
      }
    }
    F.push_back(std::move(Q));
  }
  return F;
}

template<class T>
std::vector<T> crowding_distances(const std::vector<std::vector<T>>& p, const T INF) {
  const int N = p.size();
  const int M = p.empty() ? 0 : p.front().size();
  std::vector<T> dist(p.size());
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  for(int m = 0; m < M; m++) {
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return p[i][m] < p[j][m]; });
    dist[idx.front()] = INF;
    dist[idx.back()] = INF;
    T fmin = p[idx.front()][m];
    T fmax = p[idx.back()][m];
    for(int k = 1; k + 1 < idx.size(); k++) {
      dist[idx[k]] += (p[idx[k + 1]][m] - p[idx[k - 1]][m]) / (fmax - fmin);
    }
  }
  for(int i = 0; i < dist.size(); i++) {
    if(dist[i] >= INF) dist[i] = INF;
  }
  return dist;
}

struct Individual {
  std::vector<int> x;

  Individual crossover(const Individual& p, std::mt19937& mt) const {
    static std::bernoulli_distribution dist(0.5);
    std::vector<int> next(x.size());
    for(int i = 0; i < x.size(); i++) {
      next[i] = dist(mt) ? x[i] : p.x[i];
    }
    return Individual { next };
  }

  void mutation(std::mt19937& mt) {
    static std::bernoulli_distribution dist(0.002);
    for(int i = 0; i < x.size(); i++) {
      if(dist(mt)) {
        x[i] ^= 1;
      }
    }
  }
};

struct Subject {
  std::vector<double> v;
  std::vector<double> w;
  double c;
  double sum_v;

  void build() {
    sum_v = std::accumulate(v.begin(), v.end(), 0.0);
  }

  double score(const Individual& ind) const {
    return sum_v - std::inner_product(v.begin(), v.end(), ind.x.begin(), 0.0);
  }
};

using Score = std::vector<double>;

struct MultiKnapsack {
  std::vector<Subject> subs;
  std::vector<int> fix_order;
  int N;

  void build() {
    N = subs.front().v.size();
    std::vector<double> q(subs.front().v.size(), 0);
    for(int i = 0; i < subs.size(); i++) {
      for(int j = 0; j < subs[i].v.size(); j++) {
        q[j] = std::max(q[j], subs[i].v[j] / subs[i].w[j]);
      }
    }
    fix_order.resize(q.size());
    std::iota(fix_order.begin(), fix_order.end(), 0);
    std::sort(fix_order.begin(), fix_order.end(), [&](int i, int j) { return q[i] < q[j]; });
  }

  void fix_to_feasible(Individual& ind) const {
    std::vector<double> weight(subs.size());
    std::vector<int> bad_idx;
    for(int i = 0; i < subs.size(); i++) {
      weight[i] = std::inner_product(subs[i].w.begin(), subs[i].w.end(), ind.x.begin(), 0.0);
      if(subs[i].c < weight[i]) {
        bad_idx.push_back(i);
      }
    }
    for(auto j: fix_order) {
      if(bad_idx.size() == 0) break;
      if(ind.x[j] == 0) continue;
      int k = 0;
      ind.x[j] = 0;
      while(k < bad_idx.size()) {
        int i = bad_idx[k];
        weight[i] -= subs[i].w[j];
        if(subs[i].c < weight[i]) {
          k++;
          continue;
        }
        else {
          if(k + 1 < bad_idx.size()) {
            std::swap(bad_idx[k], bad_idx.back());
          }
          bad_idx.pop_back();
        }
      }
    }
  }

  Score score(const Individual& ind) const {
    Score scores(subs.size());
    for(int i = 0; i < subs.size(); i++) {
      scores[i] = subs[i].score(ind);
    }
    return scores;
  }
};

void solve(const MultiKnapsack& knap) {
  constexpr int Cnt = 200;
  constexpr int Save = 100;
  constexpr int Gen = 20000;
  constexpr int Tounament = 2;

  std::mt19937 mt(786);

  std::vector<Individual> gen(Cnt);
  std::vector<Score> scores(Cnt);
  {
    std::uniform_int_distribution<int> dist(0, 1);
    for(int i = 0; i < Cnt; i++) {
      std::vector<int> x(knap.N);
      for(int j = 0; j < knap.N; j++) {
        x[j] = dist(mt);
      }
      gen[i].x = std::move(x);
      knap.fix_to_feasible(gen[i]);
      scores[i] = knap.score(gen[i]);
    }
  }

  for(int g = 0; g < Gen; g++) {
    auto F = non_dominated_sort(scores, Save);

    {
      double max_sum = 0;
      std::vector<double> maxs(knap.subs.size());
      for(int i = 0; i < Cnt; i++) {
        max_sum = std::max(max_sum, std::accumulate(scores[i].begin(), scores[i].end(), 0.0));
        for(int j = 0; j < scores[i].size(); j++) {
          maxs[j] = std::max(maxs[j], scores[i][j]);
        }
      }
      double sum_max = std::accumulate(maxs.begin(), maxs.end(), 0.0);
      std::cerr << g << "\t" << F[0].size() << "/" << F.size() << "\t" << max_sum << "\t" << sum_max << std::endl;
    }
    
    std::vector<Individual> next;
    std::vector<Score> next_scores;
    std::vector<int> rank;
    std::vector<double> crowd;
    for(int r = 0; r < F.size(); r++) {
      std::vector<Score> p;
      for(auto i: F[r]) {
        p.push_back(scores[i]);
      }
      auto cr = crowding_distances(p, 1e9);
      std::vector<int> idx(cr.size());
      std::iota(idx.begin(), idx.end(), 0);
      int need = std::min((int)idx.size(), std::max(0, Save - (int)next.size()));

      if(need < idx.size()) {
        std::sort(idx.begin(), idx.end(), [&](int i, int j) { return cr[i] > cr[j]; });
      }
      for(int k = 0; k < need; k++) {
        //std::cerr << cr[idx[k]] << " \n"[k + 1 == need];
        next.push_back(std::move(gen[F[r][idx[k]]]));
        next_scores.push_back(std::move(scores[F[r][idx[k]]]));
        rank.push_back(r);
        crowd.push_back(cr[idx[k]]);
      }
    }

    std::vector<int> idx(Save);
    std::iota(idx.begin(), idx.end(), 0);
    auto tounament_selection = [&]() {
      std::vector<int> vs(Tounament);
      std::sample(idx.begin(), idx.end(), vs.begin(), Tounament, mt);
      return *std::max_element(vs.begin(), vs.end(), [&](int i, int j) {
          if(rank[i] != rank[j]) return rank[i] > rank[j];
          else return crowd[i] < crowd[j]; 
      });
    };
    std::bernoulli_distribution crossover_dist(0.8);
    for(int i = Save; i < Cnt; i++) {
      if(crossover_dist(mt)) {
        int x = tounament_selection();
        int y = tounament_selection();
        //std::cerr << x << " " << y << std::endl;
        auto z = next[x].crossover(next[y], mt);
        knap.fix_to_feasible(z);
        next_scores.push_back(knap.score(z));
        next.push_back(std::move(z));
      }
      else {
        int x = tounament_selection();
        auto z = next[x];
        z.mutation(mt);
        knap.fix_to_feasible(z);
        next_scores.push_back(knap.score(z));
        next.push_back(std::move(z));
      }
    }

    std::swap(next, gen);
    std::swap(next_scores, scores);
  }

  {
    auto F = non_dominated_sort(scores, scores.size());
    std::sort(F[0].begin(), F[0].end(), [&](int i, int j) { return scores[i] < scores[j]; });
    for(auto i: F[0]) {
      for(int j = 0; j < scores[i].size(); j++) {
        std::cerr << scores[i][j] << "\t";
      }
      /*
      for(int j = 0; j < gen[i].x.size(); j++) {
        std::cerr << gen[i].x[j];
      }
      */
      std::cerr << std::endl;
    }
  }
}

int main() {
  std::mt19937 mt(81);

  MultiKnapsack multi;

  constexpr int Item = 500;
  constexpr int Subs = 3;

  for(int i = 0; i < Subs; i++) {
    Subject sub;
    std::uniform_real_distribution<double> v_dist(1, 100);
    std::uniform_real_distribution<double> c_dist(5 * Item, 10 * Item);
    sub.v.resize(Item);
    sub.w.resize(Item);
    for(int i = 0; i < Item; i++) {
      sub.v[i] = v_dist(mt);
      sub.w[i] = v_dist(mt);
    }
    sub.c = c_dist(mt);
    sub.build();
    std::cerr << sub.c << " " << sub.sum_v << " " << std::accumulate(sub.w.begin(), sub.w.end(), 0.0) << std::endl;
    multi.subs.push_back(std::move(sub));
  }
  multi.build();

  solve(multi);
}
```

{{</details>}}



