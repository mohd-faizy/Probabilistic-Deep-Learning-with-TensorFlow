[![author](https://img.shields.io/badge/author-mohd--faizy-red)](https://github.com/mohd-faizy)
![made-with-Markdown](https://img.shields.io/badge/Made%20with-markdown-blue)
![Language](https://img.shields.io/github/languages/top/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)
![Platform](https://img.shields.io/badge/platform-jupyter%20labs-blue)
![Maintained](https://img.shields.io/maintenance/yes/2024)
![Last Commit](https://img.shields.io/github/last-commit/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)
[![GitHub issues](https://img.shields.io/github/issues/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/issues)
[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://opensource.com/resources/what-open-source)
![Stars GitHub](https://img.shields.io/github/stars/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)
[![GitHub license](https://img.shields.io/github/license/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)](https://github.com/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=flat-square)](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow)
![Size](https://img.shields.io/github/repo-size/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)

# **Probabilistic Deep Learning with TensorFlow**

<img src='https://github.com/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow/blob/main/Tensorflow_Dev_png/head.png'>

> **Documentation**: [tfp_api_docs](https://www.tensorflow.org/probability/api_docs/python/tfp)

## ⁉️**Why is probabilistic programming important for deep learning?**

- The use of statistics to overcome uncertainty is one of the pillars of a large segment of the machine learning. Probabilistic reasoning has long been considered one of the foundations of inference algorithms and is represented is all major machine learning frameworks and platforms.
- Usually the classifications that you have arise and the predictions that we make, don't fall into a single category, or they fall into a category with some confidence level. Incorporating those probabilities is incredibly important for machine learning projects in the real world. Usually there is no single answer. There's this wide spectrum of answers that fall into some common distribution pattern.
- TensorFlow probability gives you the capability to take probabilistic distributions and integrate them directly with your Keras layers. TensorFlow probability despite not being part of TensorFlow Core, is an incredibly important part of the model building process.

## TensorFlow Probability is a library for probabilistic reasoning and statistical analysis.

```python
import tensorflow as tf
import tensorflow_probability as tfp # tfp is a seprate library itself

# Pretend to load synthetic data set.
features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

# Specify model.
model = tfp.glm.Bernoulli()

# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)
# ==> coeffs is approximately [1.618] (We're golden!)
```

**TensorFlow Probability (TFP)** is a Python library built on TensorFlow that makes it easy to combine probabilistic models and deep learning on modern hardware (TPU, GPU). It's for data scientists, statisticians, ML researchers, and practitioners who want to encode domain knowledge to understand data and make predictions. TFP includes:

- A wide selection of probability distributions and bijectors.
- Tools to build deep probabilistic models, including probabilistic layers and a `JointDistribution` abstraction.
- Variational inference and Markov chain Monte Carlo.
- Optimizers such as Nelder-Mead, BFGS, and SGLD

<p align='center'>
    <a href="https://youtu.be/BrwKURU-wpk" target="_blank"><img src="https://github.com/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow/blob/main/Tensorflow_Dev_png/tfp_dev_summit_ytd.jpg" height='350' width='650'></a>
</p>

> The TensorFlow Probability library provides a powerful set of tools, for statistical modeling, and makes it easy to extend our use of TensorFlow to probabilistic deep learning models. The TFP library, is part of the wider TensorFlow ecosystem, which contains a number of libraries and extensions for advanced and specialized use cases.


---

## 🎯**TensorFlow Probability (TFP) vs TensorFlow Core (TF)**  

| **Feature**                  | **TensorFlow Probability (TFP)**                                   | **TensorFlow Core (TF)**                                    |
|------------------------------|--------------------------------------------------------------------|------------------------------------------------------------|
| **Purpose**                  | Designed for probabilistic modeling and statistical inference.    | General-purpose framework for machine learning and deep learning. |
| **Focus**                    | Handles uncertainty, distributions, and Bayesian reasoning.       | Builds and trains deterministic models like neural networks. |
| **Key Components**           | Probability distributions, Bayesian layers, MCMC, variational inference. | Layers, optimizers, losses, and metrics for supervised/unsupervised learning. |
| **Distributions**            | Extensive support for probability distributions (e.g., Normal, Poisson). | Limited to simple random number generation.                |
| **Modeling Style**           | Probabilistic models with uncertainty quantification.             | Deterministic models for tasks like classification, regression, etc. |
| **Monte Carlo Support**      | Built-in tools for Monte Carlo methods and variational inference. | No dedicated Monte Carlo functionality.                    |
| **Integration**              | Seamlessly integrates with TensorFlow Core for hybrid models.    | Core framework for defining and optimizing computation graphs. |
| **Use Cases**                | Bayesian neural networks, uncertainty estimation, statistical analysis. | Supervised/unsupervised learning, reinforcement learning, deep learning. |
| **Target Audience**          | Statisticians, researchers, and Bayesian modelers.               | Machine learning practitioners and data scientists.        |

---

🎯 **Pro Tip:** Use **TensorFlow Probability** when your problem involves uncertainty, probabilistic reasoning, or Bayesian inference. For standard machine learning tasks like image classification or NLP, **TensorFlow Core** is sufficient.

---

## 🌟 Key Probability Distributions You Should Know

### 📌 Why Are These Distributions Important in TensorFlow Probability (TFP)?  

Understanding probability distributions is the foundation of modeling uncertainty in machine learning. TensorFlow Probability (TFP) provides tools to work with these distributions, enabling you to build probabilistic models, perform Bayesian inference, and generate realistic simulations. These distributions help you solve real-world problems like predicting outcomes, modeling uncertainties, and analyzing rare events.

---

### 🔷 a) [Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)  

#### **What is it?**  

A **Binomial Distribution** describes the probability of **success** or **failure** outcomes in a repeated experiment (e.g., coin flips, test results). It’s perfect for scenarios with two possible outcomes.

📝 **Key Features:**  

- **Two Outcomes:** Success (S) or Failure (F).  
- **Fixed Trials (`n`)**: The number of times the experiment is repeated.  
- **Probability (`p`)**: The likelihood of one specific outcome.  

#### **Why It's Useful in TFP:**  

Binomial distribution helps model classification problems or situations involving yes/no outcomes. Example: Predicting whether a machine fails in `n` trials.

#### Formula  

The probability of getting exactly `k` successes in `n` trials is:  
$$
f(k; n, p) = \binom{n}{k} p^k (1 - p)^{n-k}
$$  

---

### 🔷 b) [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)  

#### **What is it?**  

The **Poisson Distribution** predicts how often a rare event occurs over a specific time or space interval. For example, how many customers arrive at a store in an hour.

📝 **Key Features:**  

- Models **rare events** in a large population.  
- Events are **independent** and occur at a constant rate.  

#### **Why It's Useful in TFP:**  

Poisson distribution is used in time-series data, event modeling, or predicting rare occurrences. Example: Modeling customer arrivals or system failures.

#### Formula  

$$
f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$  
Here, `λ` (lambda) is the average number of events in a given interval.

---

### 🔷 c) [Uniform Distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)  

#### **What is it?**  

A **Uniform Distribution** is where all outcomes are equally likely. Example: Rolling a fair dice or tossing a coin.

📝 **Key Features:**  

- All values between `a` and `b` have the **same probability**.  
- Simple yet foundational for simulations and random sampling.  

#### **Why It's Useful in TFP:**  

Uniform distribution is critical for generating random variables, initializing weights in neural networks, and performing Monte Carlo simulations.

#### Formula  

Probability density function (PDF):  
$$
f(x) =
\begin{cases}
\frac{1}{b-a} & a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$  

---

### 🔷 d) [Gaussian (Normal) Distribution](https://en.wikipedia.org/wiki/Normal_distribution)  

#### **What is it?**  

The **Gaussian Distribution** (or Normal Distribution) is the famous bell curve, where most values cluster around the mean, and the probability tapers off symmetrically on both sides.

📝 **Key Features:**  

- Defined by **mean (μ)** and **standard deviation (σ)**.  
- Forms the basis of the **Central Limit Theorem**.  

#### **Why It's Useful in TFP:**  

Gaussian distributions are everywhere in machine learning, from modeling errors to designing probabilistic models. They are essential for understanding uncertainty in predictions.  

#### Formula  

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2}
$$  

---

### 🔷 e) [Exponential Distribution](https://towardsdatascience.com/what-is-exponential-distribution-7bdd08590e2a)  

#### **What is it?**  

The **Exponential Distribution** models the time until an event occurs. Example: How long until a server crashes, or the time between customer arrivals.

📝 **Key Features:**  

- Describes the **waiting time** for events.  
- Often used in reliability engineering and queueing theory.  

#### **Why It's Useful in TFP:**  

Exponential distribution helps in survival analysis and modeling waiting times in sequential processes. Example: Predicting downtime of a machine or arrival rates in traffic.  

#### Formula  

$$
f(x; \lambda) =
\begin{cases}
\lambda e^{-\lambda x} & x \geq 0 \\
0 & x < 0
\end{cases}
$$  

---

### 🚀 Summary Table of Distributions  

| Distribution      | Key Use Case                    | Formula Highlights                                              | Why Learn for TFP?                              |
|-------------------|----------------------------------|-----------------------------------------------------------------|------------------------------------------------|
| Binomial          | Success/Failure Outcomes        | $f(k; n, p) = \binom{n}{k} p^k (1-p)^{n-k}$               | Classifications, Yes/No Predictions            |
| Poisson           | Rare Events in Fixed Intervals  | $f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$        | Modeling Events over Time/Space                |
| Uniform           | Equal Likelihood of Outcomes    | $f(x) = \frac{1}{b-a}$                                    | Random Sampling, Initialization                |
| Gaussian (Normal) | Symmetric Data Distribution     | $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | Probabilistic Models, Central Limit Theorem    |
| Exponential       | Time Until Next Event           | $f(x; \lambda) = \lambda e^{-\lambda x}$                   | Survival Analysis, Sequential Event Modeling   |

🎯 **Pro Tip:** Mastering these distributions helps unlock the full potential of TensorFlow Probability, giving you a robust toolkit for solving real-world uncertainty problems.



## **Path**

<img src='Tensorflow_Dev_png\tfp-map-new.png'>

## Notes

<p align='left'>
  <a href="#"><img src='https://cdn-icons-png.flaticon.com/512/564/564445.png' width=70px height=70px alt="my_notes"></a>
</p>

:one: [:heavy_check_mark: **The TensorFlow Probability library**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/01_The%20TensorFlow_Probability_library)

:two: [:heavy_check_mark: **Probabilistic layers and Bayesian neural networks**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/02_Probabilistic_layers_and_Bayesian_Neural_Networks)

:three: [:heavy_check_mark: **Bijectors and Normalising Flows**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/03_Bijectors_and_Normalising_Flows)

:four: [:heavy_check_mark: **Variational autoencoders**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/04_Variational_Autoencoders)

:five: [:heavy_check_mark: **Capstone Project**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/05_Capstone_Project)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow&type=Date)](https://star-history.com/#mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow&Date)


## Other Resources

- [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
- [Probability Cheatsheet A](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/01_Probability_Cheatsheet_a.pdf)
- [Probability Cheatsheet B](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/02_Probability_Cheatsheet_b.pdf)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow&type=Date)](https://star-history.com/#mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow&Date)


<p align='center'>
  <a href="#"><img src='https://tymsai.netlify.app/resource/1.gif' height='10' width=100% alt="div"></a>
</p>

### $\color{skyblue}{\textbf{Connect with me:}}$

[<img align="left" src="https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/twitter_circle-512.png" width="32px"/>][twitter]
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/145/145807.png" width="32px"/>][linkedin]
<<<<<<< HEAD
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/2626/2626299.png" width="32px"/>][Portfolio]
=======
[<img align="left" src="https://cdn2.iconfinder.com/data/icons/whcompare-blue-green-web-hosting-1/425/cdn-512.png" width="32px"/>][Portfolio]
>>>>>>> 832f8a78133d6f5cbbe8f4565102ec6244eda732

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/mohd-faizy/
[Portfolio]: https://ai.stackexchange.com/users/36737/faizy?tab=profile

---

<img src="https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true" width=380px height=200px />

