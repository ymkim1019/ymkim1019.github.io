---
title: A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking (1)
date: 2018-09-12 17:09:00 +0900
mathjax: true
tags:
  - Optimization
  - Particle Filter
  - Bayesian
---

Non-Gaussian system model에서의 최적화 문제를 풀기 위해 particle filter에 대한 study를 진행한다. 다음의 링크를 참조하였다.

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
이 절에는 tracking 문제를 define한다. 시스템은 다음과 같이 모델링된다.
* $x_k$ : 시간 $k$에서의 state
* $v_k$ : 시간 $k$에서의 process noise
* $x_k=f_k(x_{k-1}, v_{k-1})$ : $f$는 이전 measurement time에서의 상태, process noise로부터의 다음 measurement time 에서의 상태로의 nonlinear mapping을 model. **(이전의 모든 상태를 알 필요 없는 order 1의 markov process)** 

그러나 system의 state를 우리는 그대로 알 수는 없고 관찰할 뿐이다.
* $z_k=h_k(x_k, n_k)$ : measurement $z_k$는 system state $x_k$와 관측 noise $n_k$로부터 얻어지며, $h_k$가 system state measurement model이다.

여기서 모든 noise는 i.i.d. 라고 가정한다.  결국 state tracking problem은 pdf $p(x_k|z_{1:k})$을 구성하는 문제이다. 이제 iterative 하게 해를 구하는 과정을 알아보자.

### Prediction
* $p(x_0|z_0)\equiv p(x_0)$ : prior
* $p(x_k|z_{1:k-1})=\int{p}(x_k|x_{k-1})p(x_{k-1}|z_{1:k-1})dx_{k-1}$ : prediction state로서 다음 state에 대한 marginal expectation 구하는 개념 ([Chapman Komogorov equation](https://en.m.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation))이며, $p(x_k|x_{k-1})$는 $f_k$로부터 얻을 수 있다. 이전 이산시간 $k-1$ 시점에서의 pdf를 이용하여 다음 이산시간 $k$ 에서의 pdf를 구한다.

### Update
이산시간 $k$ 시점의 measurement $z_k$가 얻어졌을 때, Bayes' rule로부터 prior를 update할 수 있다.

$$\begin{aligned}
p(x_k|z_{1:k})=&\dfrac{p(x_k, z_{1:k})}{p(z_{1:k})}
\\=&\dfrac{p(x_k, z_k, z_{1:k-1})}{p(z_{1:k})}
\\=&\dfrac{p(z_k|x_k, z_{1:k-1})p(x_k, z_{1:k-1})}{p(z_{1:k})}
\end{aligned}\tag{1}$$

(세번째 줄은 조건부 확률의 정의)

$z_k$와 $x_k$는 $z_{1:k-1}$로부터 독립이므로 (under markov assumption),

$$=\dfrac{p(z_k|x_k)p(x_k, z_{1:k-1})}{p(z_{1:k})}\tag{2}$$

다시 조건부 확률 정의를 적용한다.
$$\begin{aligned}
\\=&\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})p(z_{1:k-1})}{p(z_{1:k})}
\\=&\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})p(z_{1:k-1})}{p(z_k, z_{1:k-1})}
\\=&\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})p(z_{1:k-1})}{p(z_k|z_{1:k-1})p(z_{1:k-1})}
\\=&\dfrac{p(z_k|x_k)p(x_k|z_{1:k-1})}{p(z_k|z_{1:k-1})}
\end{aligned}\tag{3}$$

[수식 유도 참조 사이트](https://stats.stackexchange.com/questions/130944/deriving-the-bayes-filter-correction-equation)

여기서 분모는 normalizing constant이다. $x_k$에 대한 marginal expectation으로 구할 수 있다.

$$p(z_k|z_{1:k-1})=\int{p(z_k|x_k)p(x_k|z_{1:k-1})}dk_x\tag{4}$$

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
### SIS Particle Filter
* Monte Carlo (MC) method that forms the basis for most sequential MC filters
* Sequential MC approach is also known as bootstrap filtering, [condensation](https://endic.naver.com/enkrEntry.nhn?sLn=kr&entryId=4a70ba321a1e427e816ca55a4d7146c9&query=condensation) algorithm, particle filtering, and survial of the fittest.
* MC simulation으로 Bayesian filter를 구현하는 테크닉

그렇다면 Bayesian filter를 어떻게 구현했다는 의미인가?

* pdf : weight를 갖는 random sample로 표현
* estimate : weight를 이용하여 계산

샘플의 수가 무한대에 가까워진다면 MC method는 pdf를 완전히 표현할 수 있고, SIS filter는 optimal Bayesian estimate에 가까워진다. SIS Particle filter의 설명에 필요한 정의는 다음과 같다.

* $N_s$ : support point(particles)의 수
* $k$ : measurement time
* $x_{0:k}$ : time $k$까지의 set of states
* $\{ x^i_{0:k}\}^{N_s}_{i=1}$ : set of support points
* $w^i_k$ : k시점에서 support point $i$의 weight
* $\{ x^i_{0:k}, w^i_k \}^{N_s}_{i=1}$ : posterior pdf를 특징짓는 random measure

probability distribution으로 만들기 위해 weight의 합은 1이 되어야 한다. 시간 k에서의 posterior density는 다음과 같이 approximate될 수 있다.

$$p(x_{0:k}|z_{1:k})\approx\sum^{N_s}_{i=1}w^i_k\delta(x_{0:k}-x^i_{0:k})\tag{5}$$

$\delta(.)$는 [Dirac delta function](https://en.m.wikipedia.org/wiki/Dirac_delta_function)이며, narrow spike function을 표현하기 위해 사용된다. 신호처리 분야에서는 impulse 함수로 불리기도 한다.

$$
 \delta(x)= 
\begin{dcases}
    1,& \text{if } x = 0\\
    0,              & \text{otherwise}
\end{dcases}
\tag{6}$$

실제로 particle이 있는 point에서만 해당 particle의 weight가 pdf값이 된다(discrete weighted approximation). 무수히 paricle이 많으면 모든 point에 대해 값이 존재하고, 이 값들의 sum은 1이므로 pdf를 set of particles가 완전히 표현한다고 할 수 있다. 

#### Important Sampling
optimal solution을 찾아가기 위해서는 반복적으로 pdf를 업데이트해야 하는데, $p(x_{0:k}|z_{1:k})$로부터 어떻게 새로운 particle을 sampling 할 것인가? 

* $p(x)\propto\pi(x)$는 sample을 뽑아내기는 어렵지만 evaluate 가능한 확률밀도함수 (아래 수식전개에 $\pi(x)$가 사용되지 않는데 왜 언급되었는지?)
* $q(x)$ : $p(x)$와 달리 sample을 쉽게 추출할 수 있는 importance density이며, $x^{i}\sim{q(x)}$라고 하자.
* $w^i_k\propto\dfrac{p(x^i)}{q(x^i)}$ is the normalized weight of the $i$th particle

위의 정의로부터 다음과 같이 pdf를 근사화 할 수 있다.

$$p(x)\approx\sum^{N_s}_{i=1}w^i_k\delta(x-x^i).\tag{7}$$

만약 $x^i_{0:k}$가 $q(x_{0:k}|z_{1:k})$로부터 추출되었다면, weight는 다음과 같이 재정의 될 수 있다.

$$w^i_k\propto\dfrac{p(x^i_{0:k}|z_{1:k})}{q(x^i_{0:k}|z_{1:k})}\tag{8}$$, where $w^i$'s are  normalized.

$x$가 $q(x)$로부터 추출되었으므로 [principle of importance sampling](https://en.m.wikipedia.org/wiki/Importance_sampling)에 의해 그만큼의 확률로 weight를 나눠준다. 다시 sequential case로 돌아가보자. $k-1$ 시점에 $p(x^i_{0:k-1}|z^i_{0:k-1})$을 approximation하는 particle들이 존재할 것이고, 다음으로  새로운 sample들을 이용하여 $p(x^i_{0:k}|z^i_{0:k-1})$을 approximate해야 한다. importance density가 다음과 같이 factorize될 수 있다고 가정하자.

$$q(x_{0:k}|z_{1:k})=q(x_k|x_{0:k-1},z_{1:k})q(x_{0:k-1}|z_{1:k-1}).\tag{9}$$

그렇다면 sample $x^i_{0:k}\sim q(x_{0:k}|z_{1:k})$는 $x^i_{0:k-1}\sim q(x_{0:k-1}|z_{1:k-1})$과 $x^i_{k}\sim q(x_{0:k-1}|z_{1:k})$의 결합으로 얻어질 수 있다.

다시 아래의 식으로부터 시작해보자.

$$p(x_{0:k}|z_{1:k})=\dfrac{p(z_k|x_{0:k})p(x_{0:k}|z_{1:k-1})}{p(z_k|z_{1:k-1})}.\tag{10}$$

분모의 오른쪽 term은 다음과 같이 decompose 될 수 있다.

  $p(x_{0:k}|z_{1:k-1})=p(x_k|x_{0:k-1}|z_{1:k-1})p(x_{0:k-1}|z_{1:k-1})$

따라서,

$$\begin{aligned}
p(x_{0:k}|z_{1:k})
=&\dfrac{p(z_k|x_{0:k})p(x_k|x_{0:k-1}|z_{1:k-1})}{p(z_k|z_{1:k-1})}\\
&\times p(x_{0:k-1}|z_{1:k-1})
\end{aligned}\tag{11}.$$

다음을 (11)에 적용하면, 
* $p(z_k|x_{0:k})=p(z_k|x_k)$ : markov assumption
* $p(x_k|x_{0:k-1}|z_{1:k-1})=p(x_k|x_{k-1})$ : $z$는 state variable $x$의 변화에 영향을 주지 않음

$$=\dfrac{p(z_k|x_k)p(x_k|x_{k-1})p(x_{0:k-1}|z_{1:k-1})}{p(z_k|z_{1:k-1})}.\tag{12}$$

(9)와 (12)의 분모를 (8)에 대입하면, weight update equation은

$$\begin{aligned}
w^i_k\propto&\dfrac{p(z_k|x_k)p(x_k|x_{k-1})p(x_{0:k-1}|z_{1:k-1})}{q(x_k|x_{0:k-1},z_{1:k})q(x_{0:k-1}|z_{1:k-1})}\\
=&w^i_{k-1}\dfrac{p(z_k|x_k)p(x_k|x_{k-1})}{q(x_k|x_{0:k-1},z_{1:k})}\\
=&w^i_{k-1}\dfrac{p(z_k|x_k)p(x_k|x_{k-1})}{q(x_k|x_{k-1},z_{1:k})}
\end{aligned}\tag{13}.$$

(13)을 보면 weight update를 위해서는 바로 이전 상태인 $x_{k-1}$만 저장하면 된다. (5)를 이 관점에서 다시 정의하면,

$$p(x_{k}|z_{1:k})\approx\sum^{N_s}_{i=1}w^i_k\delta(x_{k}-x^i_{k})\tag{14}.$$

$N_s\rightarrow\infty$이면, (14)는 true posterior density에 가까워진다. SIS 알고리즘은 $q(x)$로부터 추출된 particle을 이용하여 반복적으로 weight를 업데이트하는 과정을 거친다.

다음 post에서는 SIS 알고리즘의 문제점을 살펴보고 이를 개선한 알고리즘을 다룬다.
