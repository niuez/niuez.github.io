---
title: "Genetic Algorithm + Lexicase SelectionによるBoolean CSPのアプローチ"
tags: ["Genetic Algorithm"]
date: 2022-12-29T15:00:00+09:00
---

# 概要

多目的関数の場合に線形和を取るとパラメータ調整に時間がかかってしまう。特に、目的関数がとても多い場合はもっと難しくなる。ここでは、多目的関数Genetic AlgorithmのSelection(選択)で使えるLexicase Selectionの概要と、それを用いたBoolean CSPへのアプローチをメモしておく。

今回も、目的関数は小さくするのを目標とする。

参考: [Lexicase Selection Beyond Genetic Programming](https://faculty.hampshire.edu/lspector/pubs/lexicase-beyond-gp-preprint.pdf), [スライド版](https://pdfs.semanticscholar.org/1a81/80f85842e81f1a8a85120d38ed90ca52e70b.pdf)

# Lexicase Selection

Lexicase Selectionは以下のようなアルゴリズムである。

1. $I$を現在の世代の個体全ての集合とする。
2. 目的関数をランダムにシャッフルした配列$f_i$を構成する
3. $i = 1..$について
  1. $I$を「$I$の中で$f_i$を最小にする個体の集合」に更新する。
4. $I$の中からランダムに一つ取り出し、それを親に選択する。

# Boolean CSP

CSP(制約充足問題)で変数の取る値が01になったものである。下のような式を真にするような変数の割り当てを見つける問題である。

$(\neg x_1 \lor x_3 \lor x_0) \land (x_2 \lor x_0 \lor x_4) \land (x_0 \lor x4 \lor \neg x_1)$

# multi-objective Genetic Algorithmへの落とし込み

変数はbit列で表すことができる。交叉は1点交叉や一様交叉を用いることができる。

この問題をmulti-objectiveに落とし込む方法だが、$\land$で区切られた節(Clausesと言うらしい)それぞれについて、真なら0、そうでなければ1という目的関数を作ることで構成できる。ただし、全てを分けるととてもパラメータが多くなってしまうので、いくつかの節を固めて目的関数にすることができる。この時、節の真の個数を目的関数にすることができる。

# 実装

`N`は変数の個数、`C`は制約の数、`Clauses`は目的関数で節をまとめる数、`V`はV-SATを表す。


{{< details 提出コード >}}
```cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>

using bits = std::uint64_t;

constexpr int N = 60;
constexpr int C = 50000;
constexpr int Clauses = 50;
constexpr int V = 3;

struct Individual {
  bits x = 0;

  Individual crossover(const Individual& p, std::mt19937& mt) const {
    static std::uniform_int_distribution<> idx_dist(1, N - 1);
    int i = idx_dist(mt);
    bits next = (x & ((bits(1) << i) - 1)) | (p.x & ~((bits(1) << i) - 1));
    return Individual { next };

    /*

    static std::uniform_int_distribution<bits> mask_dist(0, (bits(1) << N) - 1);
    bits mask = mask_dist(mt);
    bits next = (x & mask) | (x & ~mask);
    return Individual { next };
    */
  }

  void mutation(std::mt19937& mt) {
    constexpr double prob = 0.1;
    static std::bernoulli_distribution mut(prob);
    for(int i = 0; i < N; i++) {
      if(mut(mt)) {
        x ^= (bits(1) << i);
      }
    }
  }
};

struct Score {
  std::vector<int> val;
  int sum = 0;
};

struct Constrains {
  bits mask = 0;
  bits rev_mask = 0;

  void add(int i, int not_flag) {
    mask |= (bits(1) << i);
    rev_mask |= (bits(not_flag) << i);
  }

  int score(const Individual& ind) const {
    bits x = (ind.x & mask) ^ (rev_mask);
    if(x != 0) return 0;
    else return 1;
  }
};

struct Problem {
  std::vector<Constrains> cons;

  Score score(const Individual& ind) const {
    /*
    std::vector<int> scores(cons.size());
    int sum = 0;
    for(int i = 0; i < cons.size(); i++) {
      scores[i] = cons[i].score(ind);
      sum += scores[i];
    }
    return Score { scores, sum };
    */
    std::vector<int> scores;
    int sum = 0;
    for(int i = 0; i < cons.size(); i += Clauses) {
      int x = 0;
      for(int j = i; j < std::min((int)cons.size(), i + Clauses); j++) {
        x += cons[j].score(ind);
      }
      //int res = x == 0 ? 0 : 1;
      int res = x;
      scores.push_back(res);
      sum += res;
    }
    return Score { scores, sum };
  }
};

int lexicase_selection(const std::vector<std::pair<Score, Individual>>& gen, std::mt19937& mt) {
  std::vector<int> idx(gen.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<int> cons(gen.front().first.val.size());
  std::iota(cons.begin(), cons.end(), 0);
  std::shuffle(cons.begin(), cons.end(), mt);

  for(auto c: cons) {
    int m = 1e9;
    std::vector<int> next;
    for(auto i: idx) {
      if(m > gen[i].first.val[c]) {
        m = gen[i].first.val[c];
        next.clear();
      }
      if(m < gen[i].first.val[c]) continue;
      next.push_back(i);
    }
    idx = std::move(next);
    if(idx.size() == 1) {
      break;
    }
  }
  return idx[std::uniform_int_distribution<>(0, idx.size() - 1)(mt)];
}

void solve(const Problem& P) {
  constexpr int Count = 200;
  constexpr int Elite = 20;
  constexpr int Gen = 500;
  std::vector<std::pair<Score, Individual>> gen(Count);

  std::mt19937 mt(768);

  {
    std::uniform_int_distribution<bits> x_dist(0, (bits(1) << N) - 1);
    for(int i = 0; i < Count; i++) {
      gen[i].second.x = x_dist(mt);
      gen[i].first = P.score(gen[i].second);
      /*
      if(i == 0) {
        std::cerr << std::bitset<N>(gen[i].second.x) << std::endl;
        for(int c = 0; c < C; c++) {
          std::cerr << std::bitset<N>(gen[i].second.x & P.cons[c].mask) << std::endl;
        }
      }
      */
    }
  }

  for(int g = 0; g < Gen; g++) {
    int MIN = 1e9;
    int MAX = 0;
    int sum = 0;
    for(auto& [score, ind]: gen) {
      int s = score.sum;
      //std::cerr << s << " ";
      MIN = std::min(MIN, s);
      MAX = std::max(MAX, s);
      sum += s;
    }
    //std::cerr << std::endl;
    auto& min_ans = *std::min_element(gen.begin(), gen.end(), [](auto& a, auto& b) { return a.first.sum < b.first.sum; });
    std::cout << gen.size() << "\t" << g + 1 << "\t" << MIN << "\t" << MAX << "\t" << double(sum) / gen.size() << "\t" << std::bitset<N>(min_ans.second.x) << std::endl;
    std::vector<std::pair<Score, Individual>> next;
    {
      std::vector<int> idx(gen.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::nth_element(idx.begin(), idx.end() + Elite - 1, idx.end(), [&](int i, int j) { return gen[i].first.sum < gen[j].first.sum; });
      for(int i = 0; i < Elite; i++) {
        next.push_back(gen[idx[i]]);
        next.back().second.mutation(mt);
        next.back().first = P.score(next.back().second);
      }
    }
    for(int i = Elite; i < Count; i++) {
      int x = lexicase_selection(gen, mt);
      int y = lexicase_selection(gen, mt);
      auto ind = gen[x].second.crossover(gen[y].second, mt);
      auto score = P.score(ind);
      next.emplace_back(std::move(score), std::move(ind));
    }
    std::swap(next, gen);
  }
}

int main() {
  Problem P;
  std::mt19937 mt(81);
    
  bits ans = std::uniform_int_distribution<bits>(0, (bits(1) << N) - 1)(mt);
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);

  for(int c = 0; c < C; c++) {
    Constrains con;
    std::vector<int> sel(V);
    std::sample(idx.begin(), idx.end(), sel.begin(), V, mt);
    int ok = std::uniform_int_distribution<>(1, V)(mt);
    for(int i = 0; i < V; i++) {
      int x = (ans >> sel[i]) & 1;
      con.add(sel[i], i < ok ? 1 - x : x);
    }
    std::cerr << std::bitset<N>(con.mask) << " " << std::bitset<N>(con.rev_mask) << " " << ((con.mask & ans) ^ con.rev_mask) << std::endl;
    P.cons.push_back(con);
  }
  std::cerr << std::bitset<N>(ans) << std::endl;
  std::cin.get();
  solve(P);
}
```

{{</details>}}



