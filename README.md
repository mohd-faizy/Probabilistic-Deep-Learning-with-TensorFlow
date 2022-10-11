[![author](https://img.shields.io/badge/author-mohd--faizy-red)](https://github.com/mohd-faizy)
![made-with-Markdown](https://img.shields.io/badge/Made%20with-markdown-blue)
![Language](https://img.shields.io/github/languages/top/mohd-faizy/Probabilistic-Deep-Learning-with-TensorFlow)
![Platform](https://img.shields.io/badge/platform-jupyter%20labs-blue)
![Maintained](https://img.shields.io/maintenance/yes/2022)
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

## **Why is probabilistic programming important for deep learning?**

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

## Here are the main Distributions to have in mind

### :large_blue_diamond: a) [Binomial Distribution:](https://en.wikipedia.org/wiki/Binomial_distribution)

**What is a Binomial Distribution?**

A binomial distribution can be thought of as simply the probability of a **SUCCESS** or **FAILURE** outcome in an experiment or survey that is repeated multiple times. The binomial is a type of distribution that has two possible outcomes (the prefix “bi” means two, or twice). For example, a coin toss has only two possible outcomes: heads or tails and taking a test could have two possible outcomes: pass or fail.
what is a binomial distribution

> A Binomial Distribution shows either **(S)uccess** or **(F)ailure.**
>
> - The **first variable** in the binomial formula, `n`, stands for the number of times the experiment runs.
> - The **second variable**, `p`, represents the probability of one specific outcome.

#### **Criteria**

**Binomial distributions must also meet the following three criteria:**

1. The number of observations or trials is fixed. In other words, you can only figure out the probability of something happening if you do it a certain number of times. This is common sense—if you toss a coin once, your probability of getting a tails is 50%. If you toss a coin a 20 times, your probability of getting a tails is very, very close to 100%.

2. Each observation or trial is independent. In other words, none of your trials have an effect on the probability of the next trial.

3. The probability of success (tails, heads, fail or pass) is exactly the same from one trial to another.

Once you know that your distribution is binomial, you can apply the binomial distribution formula to calculate the probability.A Binomial Distribution shows either (S)uccess or (F)ailure.

In general, if the random variable `X` follows the binomial distribution with parameters `n ∈ ℕ` and `p ∈ [0, 1]`, we write `X ~ B(n, p)`. The probability of getting exactly `k` successes in `n` independent Bernoulli trials is given by the probability mass function:

$$
f(k, n, p)=\operatorname{Pr}(k ; n, p)=\operatorname{Pr}(X=k)=\left(\begin{array}{l}
n \\
k
\end{array}\right) p^k(1-p)^{n-k}
$$

for $k = 0, 1, 2, ..., n,$ where

$$
\left(\begin{array}{l}
n \\
k
\end{array}\right)=\frac{n !}{k !(n-k) !}
$$

is the binomial coefficient, hence the name of the distribution. The formula can be understood as follows: $k$ successes occur with probability $p^k$ and $(n − k)$ failures occur with probability $(1 − p)^{n − k}$.

### :large_blue_diamond: b) [Poisson Distribution:](https://en.wikipedia.org/wiki/Poisson_distribution)

Poisson distribution is a statistical distribution that shows how many times an event is likely to occur within a specified period of time. It is used for independent events which occur at a constant rate within a given interval of time.

> The Poisson distribution is used to describe the distribution of rare events in a large population. For example, at any particular time, there is a certain probability that a particular cell within a large population of cells will acquire a mutation. Mutation acquisition is a rare event.

#### Characteristics of a Poisson Distribution

The probability that an event occurs in a given time, distance, area, or volume is the same. Each event is independent of all other events. For example, the number of people who arrive in the first hour is independent of the number who arrive in any other hour.

The Poisson distribution is popular for modeling the number of times an event occurs in an interval of time or space.

A discrete random variable `X` is said to have a Poisson distribution with parameter `λ > 0` if for `k = 0, 1, 2, ...,` the probability mass function of X is given by

$$
f(k ; \lambda)=\operatorname{Pr}(X=k)=\frac{\lambda^k e^{-\lambda}}{k !}
$$

where

- `e` is Euler's number `(e = 2.71828...)`
- `k` is the number of occurrences
- `k!` is the factorial of `k.`

The positive real number `λ` is equal to the expected value of `X` and also to its variance.


$$
\lambda=E(X)=Var(X)
$$

The Poisson distribution can be applied to systems with a large number of possible events, each of which is rare. The number of such events that occur during a fixed time interval is, under the right circumstances, a random number with a Poisson distribution.

### :large_blue_diamond: c) [Uniform Distribution:](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)

The distribution in which all outcomes are equally likely. for example: A coin also has a uniform distribution because the probability of getting either heads or tails in a coin toss is the same.

#### The probability density function (PDF) of the continuous uniform distribution is:

$$
f(x)= \begin{cases}\frac{1}{b-a} & \text { for } a \leq x \leq b \\ 0 & \text { for } x<a \text { or } x>b\end{cases}
$$

#### The cumulative distribution function (CDF) is:

$$
F(x)= \begin{cases}0 & \text { for } x<a \\ \frac{x-a}{b-a} & \text { for } a \leq x \leq b \\ 1 & \text { for } x>b\end{cases}
$$

### :large_blue_diamond: d) [Gaussian Distribution:](https://en.wikipedia.org/wiki/Normal_distribution)

The **Gaussian Distribution**, is a probability distribution that is **Symmetric** about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a **bell curve**.

> Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known.

$$
f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$

where

$f(x)=\text{Probability density function}$

$\sigma=\text{Standard deviation}$

$\mu=\text{Mean}$

Their **Importance** is partly due to the **Central Limit Theorem**. It states that, under some conditions, the average of many samples (observations) of a random variable with finite mean and variance is itself a random variable—whose distribution converges to a normal distribution as the number of samples increases.

### :large_blue_diamond: e) [Exponential Distribution:](https://towardsdatascience.com/what-is-exponential-distribution-7bdd08590e2a)

#### Why did we have to invent Exponential Distribution?

To predict the amount of waiting time until the next event (i.e., success, failure, arrival, etc.).

**\_For example, we want to predict the following:**

- The amount of time until the customer finishes browsing and actually purchases something in your store (success).
- The amount of time until the hardware on AWS EC2 fails (failure).
- The amount of time you need to wait until the bus arrives (arrival).

$$
f(x ; \lambda)= \begin{cases}\lambda e^{-\lambda x} & x \geq 0 \\ 0 & x<0\end{cases}
$$

where:

$f(x ; \lambda)= \text{Probability Density Function}$

$\lambda=\text{Rate parameter}$

$x =\text{Random variable}$

## **Learning Path**

<img src='https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/Tensorflow_Dev_png/Probabilistic%20Deep%20Learning%20Map.png'>

## Notes

<p align='left'>
  <a href="#"><img src='https://cdn-icons-png.flaticon.com/512/564/564445.png' width=70px height=70px alt="my_notes"></a>
</p>

:one: [:heavy_check_mark: **The TensorFlow Probability library**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/01_The%20TensorFlow_Probability_library)

:two: [:heavy_check_mark: **Probabilistic layers and Bayesian neural networks**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/02_Probabilistic_layers_and_Bayesian_Neural_Networks)

:three: [:heavy_check_mark: **Bijectors and Normalising Flows**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/03_Bijectors_and_Normalising_Flows)

:four: [:heavy_check_mark: **Variational autoencoders**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/04_Variational_Autoencoders)

:five: [:heavy_check_mark: **Capstone Project**](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/tree/main/05_Capstone_Project)

## Other Resources

- [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
- [Probability Cheatsheet A](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/01_Probability_Cheatsheet_a.pdf)
- [Probability Cheatsheet B](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/02_Probability_Cheatsheet_b.pdf)

---

### Connect with me:

[<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][stackexchange ai]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/faizy-mohd-836573122/
[stackexchange ai]: https://ai.stackexchange.com/users/36737/cypher

---

![Faizy's github stats](https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true)

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=mohd-faizy&layout=compact)](https://github.com/mohd-faizy/github-readme-stats)
