---
title: A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking (2)
date: 2018-09-21 10:53:00 +0900
mathjax: true
tags:
  - Optimization
  - Particle Filter
  - Bayesian
  - SIS
  - Monte Carlo
---

[A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking (1)](https://ymkim1019.github.io/particle-filter/)에 이어서 변형 Particle Filter 알고리즘들을 살펴보도록 하자.

### 참고문헌
[A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwjJvpCLjaPdAhVJ57wKHSxeBmQQFjAAegQIABAC&url=https%3A%2F%2Fwww.irisa.fr%2Faspi%2Flegland%2Fensta%2Fref%2Farulampalam02a.pdf&usg=AOvVaw3G1TmRRCE5b7ZMpNowO18F)
[Bayesian Filtering: From Kalman Filters to Particle Filters, and Beyond](http://www2.ee.kuas.edu.tw/~lwang/WWW/BayesianFilteringFromKalmanFiltersToParticleFiltersAndBeyond.pdf)
[Auxiliary Particle Filter 관련 블로그](https://jblevins.org/notes/auxiliary-particle-filter)

## Sampling Importance Re-sampling (SIR) Filter
SIR filter와 SIS filter와의 주요한 차이점은 다음과 같다.

* prior density $p$를 importance density로 사용한다 : $q(x_k\vert x^i_{k-1}, z_{1:k})$ 대신에  $p(x_k\vert x_{k-1})$로부터 $x$를 draw
* SIR filter는 re-sampling을 항상 수행한다. 따라서 computationally more expensive하다. 
* weight degeneracy problem을 해결하는 것이 아니라 insignificant weight를 갖는 particle의 weight를 업데이트하는데 소모되는 불필요한 계산을 제거하는 것이 key idea이다.

샘플을 뽑는 과정은 다음과 같다.
* process noise $v_{k-1}\sim p_v(v_{k-1})$
* $x^i_k=f_k(x^i_{k-1},v^i_{k-1})$

weight는 다음과 같이 정의될 수 있다.
$$w^i_k\propto w^i_{k-1}p(z_k\vert x^i_k)$$
그런데 매 iteration마다 resampling 수행 후, weight을 uniform하게 설정하므로,
$$w^i_k\propto p(z_k\vert x^i_k)$$
라고 할 수 있다. SIR filter는 $z_k$를 고려하지 않고 state space를 탐색하므로 비효율적인 면이 있다. 또한 resampling을 매번 수행하므로 diversity가 빠르게 감소한다. 만약 sampling이 $z_k$를 고려하여 이루어진다면 particle filter 알고리즘은 adaptive하다고 불린다.

## Auxiliary Particle Filter
state space 탐색 시 $z_k$를 고려하지 않는 SIR filter의 단점을 보완하고자 APF가 제안되었다. 본 알고리즘에서는 good particle들을 augment하기 위한 방법으로 auxiliary variable이 추가되었다. k-1 시점의 particle index를 나타내는 auxiliary variable $i$를 포함하는 filtering distribution은 다음과 같이 정의될 수 있다.

$$\begin{aligned}
p(x_{k}, i|z_{1:k})
=&\dfrac{p(x_k, i, z_{1:k})}{p(z_{1:k})}\\
=&\dfrac{p(x_k, i, z_k, z_{1:k-1})}{p(z_{1:k})}\\
\propto &p(z_k \vert x_k, i, z_{1:k-1})p(x_k, i, z_{1:k-1})\\
=&p(z_k \vert x_k)p(x_k, i, z_{1:k-1})\\
=&p(z_k \vert x_k)p(x_k \vert i, z_{1:k-1})p(i, z_{1:k-1})\\
\propto &p(z_k \vert x_k)p(x_k \vert i, z_{1:k-1})p(i \vert z_{1:k-1})\\
=&p(z_k \vert x_k)p(x_k \vert x_{k-1}^i)p(x_{k-1}^i \vert i, z_{1:k-1})p(i \vert z_{1:k-1})\\
=&p(z_k \vert x_k)p(x_k \vert x_{k-1}^i)w^i_{k-1}\\
\end{aligned}\tag{1}.$$

(1)은 index가 $i$인 particle에서 지금까지의 measurement가 $z_{1:k}$일때, system의 state가 $x_k$일 확률을 표현한다. $\{x^j_k, i^j\}^{M_s}_{j=1}$을 샘플링하는 ($i^j$는 particle $j$의 k-1 시점 인덱스라는 의미) importance density는 다음을 만족하도록 정의된다.

$$
\begin{aligned}
q(x_{k}, i|z_{1:k})\propto p(z_k \vert u_k^i)p(x_k \vert x_{k-1}^i)w^i_{k-1}
\end{aligned}\tag{2}.
$$

$u_k^i$는 $p(x_k\vert x^i_{k-1})$와 관련된 variable이며, expectation일 수도 있고 $p(x^i_k\vert x^i_{k-1})$로부터 샘플링 될 수도 있다. $q(x_k, i\vert z_{1:k})$는 다음과 같이 factorize될 수 있다.

$$
\begin{aligned}
q(x_{k}, i|z_{1:k})=q(i \vert z_{1:k})q(x_{k}|i, z_{1:k})
\end{aligned}\tag{3}.
$$

또한 다음과 같이 정의한다.

$$
\begin{aligned}
q(x_{k}|i, z_{1:k}):=p(x_k \vert x^i_{k-1})
\end{aligned}\tag{4}.
$$

(2), (3), (4)로부터 다음이 성립한다.

$$
\begin{aligned}
q(i|z_{1:k})\propto p(z_k|u_{k}^i)w^i_{k-1}
\end{aligned}\tag{5}.
$$

weight update는 importance density를 고려해야 하므로, (1)을 (3)으로 나누면 다음과 같이 이루어진다.

$$
\begin{aligned}
w^j_k\propto w^{i^j}_{k-1}\dfrac{p(z_k \vert x^j_k)p(x^j_k \vert x^{i^j}_{k-1})}{q(x_{k}, i^j|z_{1:k})}
=\dfrac{p(z_k \vert x^j_k)}{p(z_k \vert u^{i^j}_k)}
\end{aligned}\tag{6}.
$$

알고리즘의 pseudo code는 다음과 같다.
![image](https://user-images.githubusercontent.com/25606217/45864967-ed6f6c00-bdb6-11e8-94b6-567e17455b62.PNG)
![image](https://user-images.githubusercontent.com/25606217/45864960-e7798b00-bdb6-11e8-8c42-f18173506269.PNG)

알고리즘의 흐름을 요약하면 다음과 같다. 

* 각 particle에 대해 $u_k^i$를 구한다. (e.g. $u_k^i=\mathop{\mathbb{E}}[x_{k} \vert x^i_{k-1}])$ 
* $u_k^i$를 이용하여 1차 weight를 구하고 normalize를 수행한다. 
* 1차 weight를 이용하여 k-1시점 particle에서 $N_s$개의 index를 뽑아낸다. (가장 최신의 measurement인 $z_k$가 측정될 가능성이 높은 k-1 시점의 particle들을 우선시한다는 의미) 
* 마지막으로 이렇게 선정된 k-1 시점의 particle(i.e. $x_{k-1}^i$)로부터 $x_k$를 뽑아낸다. 
* (6)을 이용하여 normalized weight를 다시 계산한다.

ASIR의 가장 큰 장점은 k-1 시점의 particle의 위치인 $x_{k-1}$로부터 $x_k$를 뽑아낼 때, 현재 measurement인 $z_k$를 고려한다는 것이다 (re-sampling을 먼저 수행하는 개념). 그러나 process noise가 큰 system이라면 $u^i_k$가 $p(x_k \vert x_{k-1})$를 잘 표한하지 못하므로, performance가 degrade된다.

## Regularized Particle Filter
Generic particle filter 알고리즘의 resampling 과정에서 diversity가 저하되는 가장 큰 이유는 sample이 discrete distribution에서 이루어지기 때문이다. Regularized Particle Filter (RPF)는 이러한 문제를 해결하는 방안 중 하나이다. RPF는 resampling 과정을 제외하면 SIS filter와 동일하다. resampling 수행 후 각 샘플에 diversity를 위한 추가적인 값을 더하여 마치 continuous distribution에서 sampling된 효과를 도모한다(regularization). 

RPF에서 샘플은 다음으로부터 추출된다.

$$
\begin{aligned}
p(x_k \vert z_{1:k}) \approx \sum^{N_s}_{i=1}w^i_kK_h(x_k-x_k^i)
\end{aligned}\tag{7}
$$

$K(\cdot)$은 다음과 같이 정의되는 rescaled kernel density이이며, symmetric하다.

$$
\begin{aligned}
K_h(x)=\dfrac{1}{h^{n_x}}K(\dfrac{x}{h})
\end{aligned}\tag{8}
$$

where $n_x$ is the dimension of the state vector $x$. 커널과 bandwidth $h$는 true posterior와 (7)사이의 오차를 최소화 하도록 선택되어야 한다. 만약 모든 sample이 동일한 weight를 갖고 있는 special case이라면, Epanechnikov Kernel이 optimal choice이다. RPF는 process noise가 작은 경우에 좋은 성능을 보인다. 알고리즘은 다음과 같다.

![image](https://user-images.githubusercontent.com/25606217/45871903-91621300-bdc9-11e8-8a54-0e5ca039e91b.png)

$D_k$는 empirical covariance matrix $S_k$에 대해 $D_kD_k^T=S_k$를 만족하는 행렬이다. $h_{opt}$는 다음과 같이 계산된다.

$$
\begin{aligned}
h_{opt}=AN^{1/(n_x+4)}_{s}
\end{aligned}\tag{9}
$$

$$
\begin{aligned}
A=[8c_{n_x}^{-1}(n_x+4)(2\sqrt{\pi})^{n_x}]^{1/(n_x+4)}
\end{aligned}\tag{10}
$$

---
다음 포스팅에서는 importance density $q()$에 대해 알아보도록 하겠다.
