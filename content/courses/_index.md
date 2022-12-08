---
title: STAT 888 Project
summary: STAT 888 project
date: "10/27/2022"

reading_time: false  # Show estimated reading time?
share: false  # Show social sharing links?
profile: false  # Show author profile?
comments: true  # Show comments?

# Optional header image (relative to `assets/media/` folder).
header:
  caption: ""
  image: ""
---
Please note that the main topic studied in this project is not about identifying a causal estimand in a specific problem, but more about one kind of methods estimating effects for time-varying treatment strategies with causal relationship.

# 1 Motivation (HW2)

Many real-world causal inference applications require us to study the effects of treatments along time, i.e., dynamic treatment. In healthcare applications, physicians apply several treatments over time, such as different drug dosage levels, types of exercise, and amount of exercise, to achieve some clinical outcomes [(Liu et al., 2020)](#2). As the action progresses, the patient's status changes accordingly. 

One the one hand, some studies on modern learning methods have been done to model and solve this type of tasks. In the field of deep learning, some sequential deep learning framework under dynamic time varying treatment strategies in complex longitudinal settings have been introduced [(Li et al., 2021)](#1). On the other hand, understanding causal relationships may help us to construct more efficient learning models, like reinforcement learning (RL), where the agent receives feedback from the environment sequentially and aim to optimize the decision policy within a given time period. 

Random variables may vary for different applications. They can be healthcare treatments and clinical outcomes along time, or promotional offers and final purchases in digital market problems. The causal effect may be average effect as function of random variables, including potential outcomes, from the set of random variables mentioned above. The hypotheses about the relationship between variables may be described in DAG, like in [(Yangyi et al., 2022)](#3):

![hw2](featured.png)

# 2 Introduction (HW2 & HW3)

In this course project, I would like to study one type of methods for estimating the effects of general dynamic treatment strategies conditioned on historical data with a generalized problem setting for patient outcomes under time-varying treatment strategies based on observed patient behaviors. In addition, I will try to understand how they can work together for dynamic treatment. The main reference for this project are Li, Rui, et al. "G-Net: a Recurrent Network Approach to G-Computation for Counterfactual Prediction Under a Dynamic Treatment Regime." [(Li et al., 2021)](#1). PMLR, 2021 and other related materials.

## 2.1 Counterfactual prediction

Counterfactual prediction	is a fundamental problem in making estimation of expected future values of variables under alternative choices of action given observed history. Counterfactual prediction is a causal task where it must account for the causal effects of treatment strategies which are different from the strategies applied.

## 2.2 G-method

Treatment strategies are usually time-varying, where decisions are made at multiple time points, and dynamic, where decisions are formed as a function of the previous history at each time point. For time-varying treatment strategies with treatment-confounder feedback, there are one kind of approaches known as “G-methods” [(Robins and Hernan, 2009)](#4) that perform well in estimating their effects. Many models have been proposed in the field of G-methods, including G-computation, structural nested models, and marginal structural models. In this project, I will focus on G-computation.

## 2.3 G-computation

G-computation is good at  estimating the effects of general dynamic treatment strategies conditioned on patient histories [(Daniel et al., 2013)](#5). A property favored by people is that the G-computation algorithm may take arbitrary regression models as input embedding model. However, introducing simple regression models in G-computation algorithm will cause limited capacity to capture complicated temporal and nonlinear causal structures.

# 3 Problem setting (HW2 & HW3)

## 3.1 G-computation for counterfactual prediction

The goal for this problem is to predict patient outcomes under various future treatment strategies given observed patient histories [(Li et al., 2021)](#1). Let:

- $t\in \lbrace 0,...,K\rbrace$: discrete time
- $A_t$: treatment action at time $t$
- $Y_t$: potential outcome at time $t$
- $L_t$: vector of covariates at time $t$ that may influence treatment decisions or be associated with the outcome
- $\overline{X_t}, \underline{X_t}$: are respectively history and future of a time-varying variable $X$
- $H_t=(\overline{L_t}, \overline{A_{t-1}})$: the patient history at but before time $t$
- $g=\lbrace g_0,...,g_K\rbrace $: dynamic treatment strategy, a collection of decision functions that map $H_t$ onto a treatment action at time $t$
- $R_t=r_t(\overline{L_t}, \overline{A_t}; \Theta)$: a representation of the history
- $r_t$: the transformation function for the history, learned by sequential learning architectures
- $\Theta$: model parameters learned during training

Therefore, $Y_t(g)$ is the counterfactual outcome observed at time $t$ had, possibly contrary to fact, given that treatment strategy $g$ been followed from baseline [(Robins, 1986)](#6). Let $Y_t(\overline{A_{m-1}}, \underline{g_m}), t>m$ denote the counterfactual outcome that would be observed if patient had received their observed treatments $\overline{A_{m-1}}$ up to time $m-1$ then followed strategy $g$ starting from time $m$. Here $g$ can be regarded as the experts.

The goal for counterfactual point prediction is to estimate expectation of counterfactual patient outcome 
$$E[Y_t(\overline{A_{m-1}}, \underline{g_m})|H_m], t \ge m$$
given observed patient history through time m for any m and any specified treatment strategy g. Another thing that we may be interested in estimating is the counterfactual outcome distributions at future time points 
$$p(Y_t(\overline{A_{m-1}}, \underline{g_m})|H_m), t \ge m$$

## 3.2 Assumptions

To estimate the expectation and distribution of counterfactual patient outcome, we need three assumptions [(Li et al., 2021)](#1):

1. Consistency: $\overline{Y_K}(\overline{A_K}) = \overline{Y_K}$. This means the observed outcome is equal to the counterfactual outcome corresponding to the observed treatment
2. Sequential Exchangeability: $\underline{Y_t}(g) \perp A_t|H_t, \forall t$. This means all confounding are observed. This would hold, e.g., if all drivers of treatment decisions that were prognostic for the outcome were observed.
3. Positivity: $P(A_t = g_t(H_t)) > 0, \forall \lbrace H_t:P(H_t) > 0\rbrace$. This means the counterfactual treatment strategy of interest has some non-zero probability of actually being followed. Positivity is not strictly necessary.

![hw3_1](hw3_1.jpg)

A causal DAG representing a time series data generating process (t=0,1) under sequential exchangeability. Note that all variables influencing treatment (i.e. with arrows directly into treatment) and associated with future outcomes are measured. [(Li et al., 2021)](#1)

## 3.3 Identification

To Help the proof, I draw a causal graph as below:

![hw3_2](hw3_2.jpg)

Under assumptions 1-3, we have the identification equality stating that the conditional distribution of the counterfactual is the conditional distribution of the observed outcome given patient history and given that treatment follows the strategy of interest [(Li et al., 2021)](#1).

For $t = m$:

$$p(Y_m(\overline{A_{m−1}}, g_m)|H_m) = p(Y_m|H_m, A_m = g_m(H_m))$$

**Proof**: This needs only a short derivation with the fact that $H_m=(\overline{L_m}, \overline{A_{m-1}})$ and $A_m=g_m(H_m)=g_m(\overline{L_m}, \overline{A_{m-1}})$ and with the help of the DAG above.

{{< math >}}
$$
\begin{aligned}
p(Y_m(\overline{A_{m-1}},g_m)|H_m)
&=p(Y_m(g_m)|H_m)\\
&=p(Y_m(A_m=g_m(H_m))|H_m)\\
&=p(Y_m|H_m, A_m=g_m(H_m))
\end{aligned}
$$
{{< /math >}}

The first equation is from causal irrelevance since $H_m$ contains all information of $A_{m-1}$ by definition. The second one is from the definition of the decision making function $g_m$. The third one is from conditional independence: all other things that have effect on $Y_m$ have been conditioned within $H_m$ except from $A_m=g_m(H_m)$. $\Box$

For $t>m$. Denote $X_{i:j} = X_i,..., X_j$ for any random variable $X$[(Li et al., 2021)](#1):

{{< math >}}
$$
\begin{aligned}
& p\left(Y_t\left(\overline{A_{m-1}}, \underline{g_m}\right)=y | H_m\right) =\int_{l_{m+1: t}} p\left(Y_t=y | H_m, L_{m+1: t}=l_{m+1: t}, A_{m: t}=g\left(H_{m: t}\right)\right)\\
& \times \prod_{j=m+1}^t p\left(L_j=l_j | H_m, L_{m+1:j-1}=l_{m+1:j-1}, A_{m, j-1}=g\left(H_m, l_{m+1: j-1}\right)\right)
\end{aligned}
$$
{{< /math >}}

**Proof**: To identify the effect in the future: $t>m$, we need to consider time-varying confounding $L$. The basic idea to derive for $t>m$ is to split it into conditional probabilities containing $L$ on each time point and take integral over $L$. Other techniques are what we used for $t=m$. 

For simplicity, we prove $t=m+1$ here:

{{< math >}}
$$
\begin{aligned}
& p\left(Y_{m+1}\left(\overline{A_{m-1}}, \underline{g_m}\right)=y | H_m\right)\\
& = p(Y_{m+1}(g_m,g_{m+1})=y|H_m)\\
& = p(Y_{m+1}=y|H_m, L_{m+1}, A_m=g(H_m), A_{m+1}=g(H_{m+1}))\\
& = \int_{l_{m+1}} p\left(Y_{m+1}=y | H_m, L_{m+1}=l_{m+1}, A_{m: m+1}=g\left(H_{m: m+1}\right)\right)\\
& \times p\left(L_{m+1}=l_{m+1} | H_m, L_m=l_m, A_{m}=g\left(H_m\right)\right)\\
& =\int_{l_{m+1}} p\left(Y_{m+1}=y | H_m, L_{m+1}=l_{m+1}, A_{m: m+1}=g\left(H_{m: m+1}\right)\right)\\
& \times p\left(L_{m+1}=l_{m+1} | H_m, A_{m}=g\left(H_m\right)\right)\\
\end{aligned}
$$
{{< /math >}}

The first equation is from causal irrelevance as in the case $t=m$ and the fact that $\underline{g_m}=(g_m, g_{m+1}$ as $t=m+1$. The second one is by g-formula and similar to the case $t=m$, except for adding a $L_{m+1}$ because the condition $H_m$ does not contain it. The third one is the expension for $L_{m+1}$ by conditional probability and integral. The last one is because $L_m$ is within $H_m$. $\Box$

This could be approximated through Monte-Carlo simulation. After simulation, assume now we have $M$ simulated draws of the counterfactual outcome for each time $t=\lbrace m, \ldots, K\rbrace$, then for each $t$, the empirical distribution constitutes a Monte-Carlo approximation of the counterfactual outcome distribution. The sample averages at time $t$ are an estimate of the conditional expectations and can serve as point predictions for $Y_t\left(\overline{A_{m-1}}, \underline{g_m}\right)$ in a patient with history $H_m$ [(Li et al., 2021)](#1).

# 4 Estimation (HW4)

## 4.1 G-computation algorithm 

The estimation is delevoped under the general g-computation algorithm, described as following:

{{< math >}}
$$
\begin{aligned}
&\begin{aligned}
&\hline \text { Algorithm } 1 \text { G-Computation (One simulation) } \\
&\hline \text { Set } a_m^*=g_m\left(H_m\right) \\
&\text { Simulate } l_{m+1}^* \text { from } p\left(L_{m+1} \mid H_m, A_m=a_m^*\right) \\
&\text { Set } a_{m+1}^*=g_m\left(H_m, l_{m+1}^*, a_m^*\right) \\
&\text { Simulate } l_{m+2}^* \quad \text { from } \quad p\left(L_{m+2} \mid H_m, L_{m+1}=\right.\left.l_{m+1}^*, A_m=a_m^*, A_{m+1}=a_{m+1}^*\right)
\end{aligned}\\
&\text { Continue simulations through time } K \\
\end{aligned}
$$
{{< /math >}}

Note that in the following discussion, $Y_t$ are regarded as one of the covariates in $L$.

## 4.2 G-Net

An important step in the g-computation algorithm is to simulate $p(L_t|\overline{L_{t-1}}, \overline{A_{t-1}})$ of the covariates from joint conditional distributions given history at time $t$. In practice, this needs to be estimated from data since we do not have the true conditional distributions. Generalized linear regression models are always used to estimate the conditional distributions of the covariates, while these models are in lack of the ability to capture temporal dependencies embedded in data. G-Net is proposed by [(Li et al., 2021)](#1) for this task.

G-Net framework makes use of sequential deep learning models to estimate conditional distributions $p(L_t|\overline{L_{t-1}}, \overline{A_{t-1}})$ and then simulate variables under treatment strategies $g$. In practice, G-Net divides the components of the covariates into several batches since customizing individual simulation models for components may often perform better and be easy to implement. To be specific, denote $L_t^0,..., L_t^{p-1}$ be the $p$ batches of $L_t$ by an arbitrary preceding order. For each simulation, there is the basic probability identity [(Li et al., 2021)](#1), from which G-Net will simulate the left hand side probability by simulating each $L_t^j$ on the right hand side by the given order:

{{< math >}}
$$
\begin{aligned}
p(L_t|\overline{L_{t-1}}, \overline{A_{t-1}}) 
&= p(L_t^0|\overline{L_{t-1}}, \overline{A_{t-1}}) \times p(L_t^1|L_t^0, \overline{L_{t-1}}, \overline{A_{t-1}}) \\
& \times ... \times p(L_t^{p-1}|L_t^0,..., L_t^{p-2}, \overline{L_{t-1}}, \overline{A_{t-1}})
\end{aligned}
$$
{{< /math >}}

The next step is to estimate conditional expectations $E[L_t^j|L_t^0,..., L_t^{j-1}, \overline{L_{t-1}}, \overline{A_{t-1}}]$. Different methods are applied to parametric or non-parametric situation:
- Under parametric assumption, estimation of the distribution can be done by simply maximizing the likelihood, and sampling methods may vary according to the distribution family. 
- Under non-parametric assumption, estimation can be done by an empirical way: simulate from $L_t^j|L_t^0,..., L_t^{j-1}, \overline{L_{t-1}}, \overline{A_{t-1}}\sim \hat E[L_t^j|L_t^0,..., L_t^{j-1}, \overline{L_{t-1}}, \overline{A_{t-1}}] +\epsilon_t^j$, where $\epsilon_t^j$ is draw from an empirical distribution of $L_t^j-\hat L_t^j$ in a holdout set which has not been used. 

After estimating the conditional expectations, estimates for covariate components can be obtained by the conditional expectations of covariates. To simplify the notation, based on the representation of the history: $R_t$, the conditional expectation of each $L^j_{t+1}, 0 ≤ j < p$ can be denoted as being estimated by the specialized estimation functions $f^j_t$:

{{< math >}}
$$
\begin{aligned}
L_{t+1}^0 &= f_t^0(R_t;\Lambda_0)\\
... &\\
L_{t+1}^j &= f_t^j(R_t, L_{t+1}^0,..., L_{t+1}^{j-1};\Lambda_j)
\end{aligned}
$$
{{< /math >}}

where $f_t^j, \Lambda_j$ are learnable parameters. Note that The $f^j_t$ might be sequential models (e.g. RNN) or models focused only on the representation at t (e.g. linear models)[(Li et al., 2021)](#1).

In G-Net, [(Li et al., 2021)](#1) uses teacher-forcing by [(Williams and Zipser, 1989)](#7) to learn the parameters $(\Theta, \Lambda)$ for estimating covariates $L_t$. With parameters determined, the distribution of the Monte Carlo simulations produced by the algorithm in 4.1 constitute an estimate of uncertainty about a counterfactual prediction. The whole process is depicted as following:

![hw4_1](hw4_1.png)

The G-Net: A flexible sequential deep learning framework for g-computation [(Li et al., 2021)](#1)

# 5 Experiment (HW4)

[(Li et al., 2021)](#1) performed experiments on data generated by CVSim, a program that simulates the dynamics of the human cardiovascular system [(Heldt et al., 2010)](#8).

## 5.1 Data Generation

The data contains two parts:

- An ‘observational’ dataset $D_o$ (past patients) is under treatment regime $g_o$ (usual care).
- Two ‘counterfactual’ datasets $D_{c1}$ and $D_{c2}$ (ground truth data for a collection of patients whose trajectories follow the path we are interested in predicting) are under treatment regimes $g_{c1}$ and $g_{c2}$ (different treatment strategy of interest). They have the same generating process except for the treatment. 

## 5.2 Experimental Setup

The task is training G-Net on the observational regime dataset $D_o$ and using it to predict the trajectories of patients in the counterfactual regime dataset $D_{cj}$ for time steps $m$ to $K$ for each $j$.

In the experiment there are four situations. In the table below, rows represent different choices of estimation function $f$ and columns represent whether the model is based on the entire history or only the current time point.

![hw4_2](hw4_2.png)

There are a total of 12000 simulated trajectories generated from CVSim in $D_o$ ($N_o = 12000$), of which 83% (10000) were used for training, and the remaining 17% (2000) for validation. For testing, there are 1000 simulated trajectories in the $D_{cj}$ datasets ($N_c = 851$ after filtering). 

## 5.3 Result & Discussion

- MLPs (which capture nonlinearity but not long term temporal dependency) did not outperform GLMs, which suggests that much of the gain in performance from LSTM-based implementations comes from incorporation of patient history. 
- G-Net performs over two other state-of-the-art RNN based techniques, rMSN and CRN.
- While using flexible regression models, we should be careful about extrapolating to counterfactual treatment strategies too far afield from what appears in the data. 
- G-Net is not expected to outperform alternatives when there are no nonlinear or long term dependencies in the data
- The estimate for a counterfactual prediction by G-Net ignores uncertainty about the G-Net parameter estimates themselves, which may be approximated by fitting a Bayesian model.

# 6 Sensitivity Analysis (HW5)

Although usual sensitivity analysis: Manski's, Rosenbaum's, and VanderWeele's approaches were learned in class, this G-Net method is not typically an object to apply sensitivity analysis.

## 6.1 Assumptions review

Recall the assumptions for the G-Net are:

1. Consistency: $\overline{Y_K}(\overline{A_K}) = \overline{Y_K}$. This means the observed outcome is equal to the counterfactual outcome corresponding to the observed treatment.
2. Sequential Exchangeability: $\underline{Y_t}(g) \perp A_t|H_t, \forall t$. This means all confounding are observed. This would hold, e.g., if all drivers of treatment decisions that were prognostic for the outcome were observed.
3. Positivity: $P(A_t = g_t(H_t)) > 0, \forall \lbrace H_t:P(H_t) > 0\rbrace$. This means the counterfactual treatment strategy of interest has some non-zero probability of actually being followed. Positivity is not strictly necessary.

# References

<div id ="1"></div>

- [1] [G-Net: a Recurrent Network Approach to G-Computation for Counterfactual Prediction Under a Dynamic Treatment Regime](https://proceedings.mlr.press/v158/li21a)

<div id ="2"></div>

- [2] [Reinforcement learning for clinical decision support in critical care](https://www.jmir.org/2020/7/e18477/)

<div id ="3"></div>

- [3] [Efficient Reinforcement Learning with Prior Causal Knowledge](https://proceedings.mlr.press/v177/lu22a/lu22a.pdf)

<div id ="4"></div>

- [4] [Estimation of the causal effects of time varying exposures. In Garrett Fitzmaurice, Marie Davidian, Geert Verbeke, and Geert Molenberghs, editors, Longitudinal Data Analysis](https://cdn1.sph.harvard.edu/wp-content/uploads/sites/343/2013/03/abc.pdf)

<div id ="5"></div>

- [5] [Methods for dealing with time‐dependent confounding](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.5686?casa_token=d1IB83DPXvYAAAAA%3AtDUKy3FwHs4XAX_p-rbqMpPYVWsUBTHigJHHuvsIUAjihDQG49F4us8yFAUGzEHkQ1K_NgqVdOcLg5u2_A)

<div id ="6"></div>

- [6] [A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect](https://www.sciencedirect.com/science/article/pii/0270025586900886)

<div id ="7"></div>

- [7] [A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](https://direct.mit.edu/neco/article-abstract/1/2/270/5490/A-Learning-Algorithm-for-Continually-Running-Fully)

<div id ="8"></div>

- [8] [CVSim: An Open-Source Cardiovascular Simulator for Teaching and Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3178445/)






