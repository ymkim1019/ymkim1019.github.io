---
title: A Tutorial on Particle Filters
date: 2018-09-06 14:00:00 +0900
tags:
  - Optimization
  - Particle Filter
  - Bayesian
---

# A Tutorial on Particle Filters

test eq
$a_b$

Non-gaussian model에서의 최적화 문제를 풀기 위해 particle filter에 대한 study를 진행한다. 다음의 링크를 참조하였다.

[A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwjJvpCLjaPdAhVJ57wKHSxeBmQQFjAAegQIABAC&url=https%3A%2F%2Fwww.irisa.fr%2Faspi%2Flegland%2Fensta%2Fref%2Farulampalam02a.pdf&usg=AOvVaw3G1TmRRCE5b7ZMpNowO18F)

* vector의 경우 bold로 표현하는 것이 일반적이나 입력의 편의성을 위해 본 문서에서는 bold처리하지 않음


## Introduction
### 배경
* Physical system을 정확하게 모델링하기 위해서는 nonlinearity, non-Gaussianity를 고려해야 함
* 또한 online 처리가 고려되어야 함 (이전의 모든 데이터를 저장하지 않아도 새로운 데이터의 처리가 가능해야 함)
* state vector가 system에 대한 정보를 담고 있다고 가정
* 그러나 measurement vector는 noisy함 (represent noisy observations)

### Why Bayesian Approach?
시스템을 추론하기 위해서는 최소한 다음 두 가지의 model이 필요하다.
* 시간에 따른 system의 evolution을 표현하는 model
* noisy measurement와 state 사이의 관계를 표현하는 model

이러한 model이 probabilistic form으로 존재한다고 가정한다. Bayesian approach는 model을 확률적으로 표현하고, new observation에 대한 즉각적인 반영이 가능하다는 점에서 위에서 언급한 요구사항을 가장 잘 만족하는 접근 방법이라고 할 수 있다. 
시간의 흐름에 따라 새로운 측정값을 얻는 문제의 경우에는 일반적으로 recursive filter를 사용한다. recursive filter는 다음의 두 stage로 구분된다.
* prediction : 다음 meaurement time에서의 system state를 pdf를 가지고 prediction
* update : latest measurement를 가지고 prediction pdf를 update (Bayes theorem을 이용)

## Non-linear Bayesian Tracking
이 절에는 tracking 문제를 define한다. 
* $$x_k$$ : 시간 $k$에서의 state
* $v_k$ : 시간 $k$에서의 process noise
* $x_k=f_k(x_{k-1}, v_{k-1})$ : $f$는 이전 measurement time에서의 상태, process noise로부터의 다음 measurement time 에서의 상태로의 nonlinear mapping을 model. **(order 1의 markov prodess)** #1

그러나 system의 state를 우리는 직접적으로 관찰할 수 있는 것이 아니다.
* $z_k=h_k(x_k, n_k)$ : measurement $z_k$는 system state $x_k$와 관측 noise $n_k$로부터 얻어지며, $h_k$가 system state 측정을 model하는 function이다.

여기서 모든 noise는 i.i.d. 라고 가정한다.  결국 tracking problem은 pdf $p(x_k|z_{1:k})$을 구성하는 문제이다.

### Prediction
* $p(x_0|z_0)\equiv p(x_0)$ : prior
* $p(x_k|z_{1:k-1})=\int{p}(x_k|x_{k-1})p(x_{k-1}|z_{1:k-1})dx_{k-1}$ : prediction state로서 다음 state에 대한 marginal expectation 구하는 개념 ([Chapman Komogorov equation](https://en.m.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation))이며, $p(x_k|x_{k-1})$는 $f_k$로부터 얻을 수 있다.

### Update
Bayes' rule로부터 prior를 update할 수 있다.

$p(x_k|z_{1:k})\\
=\dfrac{p(x_k, z_{1:k})}{p(z_{1:k})}\\
=\dfrac{p(x_k, z_k, z_{1:k-1})}{p(z_{1:k})}$

조건부 확률의 정의로부터 다음의 식이 유도된다.

$=\dfrac{p(z_k|x_k, z_{1:k-1})p(x_k, z_{1:k-1})}{p(z_{1:k})}$

$z_k$와 $x_k$는 $z_{1:k-1}$로부터 독립이므로 (markov),

$=\dfrac{p(z_k|x_k)p(x_k, z_{1:k-1})}{p(z_{1:k})}$

다시 조건부 확률 정의를 적용한다.

$=\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})p(z_{1:k-1})}{p(z_{1:k})}$

$=\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})p(z_{1:k-1})}{p(z_k, z_{1:k-1})}$

$=\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})p(z_{1:k-1})}{p(z_k|z_{1:k-1})p(z_{1:k-1})}$

$=\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})}{p(z_k|z_{1:k-1})}$

[수식 유도 참조 사이트](https://stats.stackexchange.com/questions/130944/deriving-the-bayes-filter-correction-equation)

여기서 분모는 normalizing constant이다. $x_k$에 대한 marginal expectation으로 구할 수 있다.

$p(z_k|z_{1:k-1})=\int{p(z_k|x_k)p(x_k|z_{1:k-1})}dk_x$

그러나 이러한 recursive propagation은 restrictive case(예: Kalman filter, grid-based filter)에서만 analytically 결정될 수 있다.

## Optimal Algorithms
아래 두 알고리즘은 본 문서가 다루고자 하는 범위 밖의 내용이므로 설명을 생략한다.
* Kalman FIlter
* Grid-Based Algorithm

## Suboptimal Algorithms
많은 실제 문제에서는 optimal algorithms들이 정의한 가정들이 적절하지 않다. 이러한 경우에는 approximation이 요구된다. 다음 알고리즘들은 널리 사용되는 nonlinear Bayesian filter들이다.

* EKF
* approximate grid-based method
* particle filter

위 알고리즘 중, 본 문서에서는 particle filter에 대해 중점적으로 다룬다.

## Particle Filtering Methods
