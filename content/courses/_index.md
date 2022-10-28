---
title: STAT 888 Project
summary: STAT 888 project
date: "10/5/2022"

reading_time: false  # Show estimated reading time?
share: false  # Show social sharing links?
profile: false  # Show author profile?
comments: true  # Show comments?

# Optional header image (relative to `assets/media/` folder).
header: false
  caption: ""
  image: ""
---
Please note that the main topic studied in this project is not about identifying a causal estimand in a specific problem, but more about one kind of methods estimating effects for time-varying treatment strategies with causal relationship.

# Motivation

Many real-world causal inference applications require us to study the effects of treatments along time, i.e., dynamic treatment. In healthcare applications, physicians apply several treatments over time, such as different drug dosage levels, types of exercise, and amount of exercise, to achieve some clinical outcomes (Liu et al., 2020). As the action progresses, the patient's status changes accordingly. 

One the one hand, some studies on modern learning methods have been done to model and solve this type of tasks. In the field of deep learning, some sequential deep learning framework under dynamic time varying treatment strategies in complex longitudinal settings have been introduced (Rui Li et al., 2021). On the other hand, understanding causal relationships may help us to construct more efficient learning models (Yangyi Le et al., 2022), like reinforcement learning (RL), where the agent receives feedback from the environment sequentially and aim to optimize the decision policy within a given time period. 

Random variables may vary for different applications. They can be healthcare treatments and clinical outcomes along time, or promotional offers and final purchases in digital market problems. The causal effect may be average effect as function of random variables, including potential outcomes, from the set of random variables mentioned above. The hypotheses about the relationship between variables may be described in DAG, like in Yangyi Le et al. (2022):

![hw2](featured.png)

# Introduction

In this course project, I would like to study one type of methods for estimating the effects of general dynamic treatment strategies conditioned on historical data. In addition, I will try to understand how they can work together for dynamic treatment. The main reference for this project are Li, Rui, et al. "G-Net: a Recurrent Network Approach to G-Computation for Counterfactual Prediction Under a Dynamic Treatment Regime." [Li et al. (2021)](#1). PMLR, 2021 and other related materials.

## g-method

Treatment strategies are usually time-varying, where decisions are made at multiple time points, and dynamic, where decisions are formed as a function of the previous history at each time point. For time-varying treatment strategies with treatment-confounder feedback, there are one kind of approaches known as “G-methods” (Hernan and Robins, 2020; Robins and Hernan, 2009) that perform well in estimating their effects. Many models have been proposed in the field of G-methods, including g-computation (Robins, 1986, 1987), structural nested models (Robins, 1994; Vansteelandt and Joffe, 2014), and marginal structural models (Robins et al., 2000; Orellana et al., 2008). In this project, I will focus on g-computation.

## g-computation

g-computation is good at  estimating the effects of general dynamic treatment strategies conditioned on patient histories (Daniel et al., 2013).


$A_t$: treatment


# References

<div id ="1"></div>

- [1] [G-Net: a Recurrent Network Approach to G-Computation for Counterfactual Prediction Under a Dynamic Treatment Regime](https://proceedings.mlr.press/v158/li21a)