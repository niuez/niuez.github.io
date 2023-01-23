---
title: "深層強化学習とREINFORCE algorithmによるCartPoleのアプローチ(強化学習第2版 13 REINFORCE algorithm)"
tags: ["Reinforcement Learning"]
date: 2023-01-22T15:00:00+09:00
---

# 深層強化学習へ

これまでの強化学習の勉強では、状態空間が配列に収まるようなものを見てきた。しかし、実数を取るものや、もっと状態空間が広いものに対応するため、深層強化学習に手をつけることにした。

[つくりながら学ぶ！深層強化学習 PyTorchによる実践プログラミング - Amazon](https://amzn.asia/d/7gizeeg)を買って勉強することにした。半分くらいは普通の強化学習を扱うので、強化学習を一から始める人にはおすすめ。自分はその半分を強化学習第2版で埋めてたので、買わなくてよかったかも...

# CartPole

OpenAIのgymにある[CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)という例題を用いた。カートを左右に動かして、ポールを倒さないようにするのが目的である。

# PyTorchを用いて実装したものの...

上の本を参考にDQNで解くことはできた。しかし、PyTorchや深層学習への経験が浅く、何がしたくてそのコードを書いているかわからず、それが原因でREINFORCE algorithmに改造する方法もわからなかったので、PyTorchのREINFORCE algorithmを参考にしつつ、一回全部C++で書くことにした(？)。

# PyTorchによるREINFORCE algorithm

[reinforcement\_learning/REINFORCE\_cartpole.ipynb at main · niuez/reinforcement\_learning](https://github.com/niuez/reinforcement_learning/blob/main/cartpole/REINFORCE_cartpole.ipynb)

PyTorchが何をしてるのかわからないので、これをC++で全部フルスクラッチします。

# CartPoleをC++へ移植

[gym/cartpole.py at master · openai/gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)で公開されているので、これをそのまま移植します。

# NeuralNetworkの構築

## 初期化

- 入力層 4
- 全結合層 32: 重み+バイアスの`4*32 + 32`パラメータ
- 活性化層 32: ReLU
- 全結合層 2: 重み+バイアスの`32 * 2 + 2`パラメータ
- 出力層 2: softmax、$\pi(a \| s, \theta)$にあたる。

パラメータは[【機械学習】パラメータの重みの初期値 - Qiita](https://qiita.com/m-hayashi/items/02065a2e2ec3e2269e0b)を参考にしてHeの初期値を用いた。

`value`が各ノードの値、`theta`がパラメータである。

```cpp
struct NN {
  using value_type = double;
  std::vector<std::vector<value_type>> value;
  std::vector<std::vector<value_type>> theta;

  template<class URGB>
  NN(URGB& g) {
    value.resize(5);
    theta.resize(4);

    value[0].resize(4);
    value[1].resize(32);
    value[2].resize(32);
    value[3].resize(2);
    value[4].resize(2);

    theta[0].resize(4 * 32 + 32);
    theta[2].resize(32 * 2 + 2);

    for(int i = 0; i < theta.size(); i++) {
      for(int j = 0; j < theta[i].size(); j++) {
        if((i == 0 && j % 5 == 4) || (i == 2 && j % 33 == 32)) {
          theta[i][j] = std::uniform_real_distribution<value_type>(- 1.0 / value[i].size(), 1.0 / value[i].size())(g);
        }
        else {
          theta[i][j] = std::normal_distribution<value_type>(0, std::sqrt(2.0 / value[i].size()))(g);
        }
      }
    }
  }
  ...
};
```

## 評価

いわゆるforward操作だと思う。各層の定義に従って、下から計算する。

```cpp
  void evaluate() {
    {
      for(int i = 0; i < value[1].size(); i++) {
        value[1][i] = 0;
        for(int j = 0; j < value[0].size(); j++) {
          value[1][i] += value[0][j] * theta[0][i * (value[0].size() + 1) + j];
        }
        value[1][i] += theta[0][i * (value[0].size() + 1) + value[0].size()];
      }
    }
    {
      for(int i = 0; i < value[1].size(); i++) {
        value[2][i] = std::max(value[1][i], value_type(0));
      }
    }
    {
      for(int i = 0; i < value[3].size(); i++) {
        value[3][i] = 0;
        for(int j = 0; j < value[2].size(); j++) {
          value[3][i] += value[2][j] * theta[2][i * (value[2].size() + 1) + j];
        }
        value[3][i] += theta[2][i * (value[2].size() + 1) + value[2].size()];
      }
    }
    {
      int min_i = std::min_element(value[3].begin(), value[3].end()) - value[3].begin();
      if(value[3][1 - min_i] - value[3][min_i] >= 30) {
        value[4][min_i] = 0;
        value[4][1 - min_i] = 1;
      }
      else {
        float sigma = 0;
        for(int i = 0; i < value[3].size(); i++) {
          sigma += std::exp(value[3][i] - value[3][min_i]);
        }
        for(int i = 0; i < value[3].size(); i++) {
          value[4][i] = std::exp(value[3][i] - value[3][min_i]) / sigma;
        }
      }
    }
  }
```

## $\nabla \log \pi(a \| s, \theta)$の計算

evaluateした後に、後ろ向きに自動微分を行う。[自動微分を実装して理解する（後編） - Qiita](https://qiita.com/lotz/items/f1d4ab1d83dc13a5d81a)

$$\frac{\partial \log \mathrm{softmax}_1(x_1, x_2)}{\partial x_1} = 1 - \mathrm{softmax}(x_1)$$
$$\frac{\partial \log \mathrm{softmax}_1(x_1, x_2)}{\partial x_2} = - \mathrm{softmax}(x_2)$$

に気を付ける

```cpp
  std::vector<std::vector<value_type>> log_gradient(int out_i) const {
    std::vector<std::vector<value_type>> g_va = value;
    std::vector<std::vector<value_type>> g_th = theta;

    g_va[4][out_i] = 1;

    {
      for(int i = 0; i < g_va[3].size(); i++) {
        if(i == out_i) {
          g_va[3][i] = (1 - value[4][i]);
        }
        else {
          g_va[3][i] = -value[4][i];
        }
      }
    }
    {
      for(int j = 0; j < value[2].size(); j++) {
        g_va[2][j] = 0;
      }
      for(int i = 0; i < value[3].size(); i++) {
        for(int j = 0; j < value[2].size(); j++) {
          g_th[2][i * (value[2].size() + 1) + j] = g_va[3][i] * value[2][j];
          g_va[2][j] += g_va[3][i] * theta[2][i * (value[2].size() + 1) + j];
        }
        g_th[2][i * (value[2].size() + 1) + value[2].size()] = g_va[3][i];
      }
    }
    {
      for(int i = 0; i < value[1].size(); i++) {
        g_va[1][i] = g_va[2][i] * (value[1][i] <= 0 ? 0 : 1);
      }
    }
    {
      for(int j = 0; j < value[0].size(); j++) {
        g_va[0][j] = 0;
      }
      for(int i = 0; i < value[1].size(); i++) {
        for(int j = 0; j < value[0].size(); j++) {
          g_th[0][i * (value[0].size() + 1) + j] = g_va[1][i] * value[0][j];
          g_va[0][j] += g_va[1][i] * theta[0][i * (value[0].size() + 1) + j];
        }
        g_th[0][i * (value[0].size() + 1) + value[0].size()] = g_va[1][i];
      }
    }

    return g_th;
  }
```

## ADAM

[Adam — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)を参考に実装する。

```cpp
  std::vector<std::vector<value_type>> mt;
  std::vector<std::vector<value_type>> vt;
  double bp1 = 1;
  double bp2 = 1;

  void adam(const std::vector<std::vector<value_type>>& th, const value_type& G, const value_type lr) {
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    if(mt.empty()) {
      mt = th;
      vt = th;
      bp1 = 1;
      bp2 = 1;
      for(int i = 0; i < th.size(); i++) {
        for(int j = 0; j < th[i].size(); j++) {
          mt[i][j] = vt[i][j] = 0;
        }
      }
    }
    bp1 *= beta1;
    bp2 *= beta2;

    for(int i = 0; i < th.size(); i++) {
      for(int j = 0; j < th[i].size(); j++) {
        mt[i][j] = beta1 * mt[i][j] + (1 - beta1) * th[i][j] * G;
        vt[i][j] = beta2 * vt[i][j] + (1 - beta2) * std::pow(th[i][j] * G, 2);
        theta[i][j] -= lr * mt[i][j] / (1 - bp1) / (std::sqrt(vt[i][j] / (1 - bp2)) + 1e-9);
      }
    }
  }
```

# 学習の本体

```cpp
int main() {
  constexpr int MAX_TRY = 500;
  constexpr int MAX_STEP = 200;
  constexpr double Gamma = 0.99;
  using value_type = NN::value_type;
  CartPole_v1 state;
  std::mt19937 mt(788);

  NN nn(mt);

  int ok = 0;

  for(int ti = 0; ti < MAX_TRY; ti++) {
    state.reset(mt);
    int t = 0;
    std::vector<value_type> rs;
    std::vector<std::vector<std::vector<value_type>>> gs;
    for(; t < MAX_STEP; t++) {
      nn.value[0][0] = state.state.cart_pos;
      nn.value[0][1] = state.state.cart_vel;
      nn.value[0][2] = state.state.pole_pos;
      nn.value[0][3] = state.state.pole_vel;
      nn.evaluate();
      int action = std::bernoulli_distribution(nn.value[4][0])(mt) ? 0 : 1;
      //std::cerr << state.state << " " << nn.value[4][0] << " " << nn.value[4][1] << " " << action << std::endl;
      gs.push_back(nn.log_gradient(action));
      bool is_saved = state.step(action) && (t + 1) < MAX_STEP;
      float reward = 0;
      if(!is_saved) {
        reward = t < MAX_STEP - 5 ? -1 : 1;
      }
      rs.push_back(reward);
      if(!is_saved) {
        break;
      }
    }
    //std::cerr << state.state << std::endl;
    std::cerr << ti << "\t:" << t << std::endl;
    if(t + 1 == MAX_STEP) {
      ok++;
      if(ok == 10) {
        std::cerr << "10 consecutive successes" << std::endl;
        break;
      }
    }
    else {
      ok = 0;
    }

    double G = 0;
    std::vector<double> Gs(rs.size());
    double G_sum = 0;
    double G_s2 = 0;
    for(int i = rs.size(); i --> 0;) {
      G = G * Gamma + rs[i];
      Gs[i] = G;
      G_sum += G;
      G_s2 += G * G;
    }
    double mean = G_sum / rs.size();
    double stddev = std::sqrt(G_s2 / rs.size() - mean * mean);

    std::vector<std::vector<value_type>> delta;
    for(int i = rs.size(); i --> 0;) {
      Gs[i] = (Gs[i] - mean) / (stddev + 1e-9);
      if(delta.empty()) {
        delta = gs[i];
        for(int x = 0; x < gs[i].size(); x++) {
          for(int y = 0; y < gs[i][x].size(); y++) {
            delta[x][y] = gs[i][x][y] * -Gs[i];
          }
        }
      }
      else {
        for(int x = 0; x < gs[i].size(); x++) {
          for(int y = 0; y < gs[i][x].size(); y++) {
            delta[x][y] += gs[i][x][y] * -Gs[i];
          }
        }
      }
    }
    nn.adam(delta, 1, 1e-2); // t=201で200stepを連続10回出せた
  }
}
```

できた。

[コードの全体](https://github.com/niuez/reinforcement_learning/blob/main/cartpole/nn.cpp)
