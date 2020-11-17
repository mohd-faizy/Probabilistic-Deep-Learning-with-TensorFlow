# **Probabilistic Deep Learning with TensorFlow**

<img src='https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/Tensorflow_Dev_png/head.png'>

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

[![Watch the video](https://img.youtube.com/vi/BrwKURU-wpk/maxresdefault.jpg)](https://youtu.be/BrwKURU-wpk)

> The TensorFlow Probability library provides a powerful set of tools, for statistical modeling, and makes it easy to extend our use of TensorFlow to probabilistic deep learning models. The TFP library, is part of the wider TensorFlow ecosystem, which contains a number of libraries and extensions for advanced and specialized use cases.

---

# :zero::one: **The TensorFlow Probability library**
---

## :black_circle: **a) Univariate distributions**

**Distribution objects** are vital building blocks to build probabilistic deep learning models as these objects capture the essential operations on probability distributions that we're going to need as we build these models.

## :black_circle: **b) Multivariate distributions**

## :black_circle: **c) The Independent distribution**

## :black_circle: **d) Sampling and log probs**

## :black_circle: **e) Trainable distributions**

---

# :zero::two: **Probabilistic layers and Bayesian neural networks**

---


### Connect with me:


[<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][StackExchange AI]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/faizy-mohd-836573122/
[StackExchange AI]: https://ai.stackexchange.com/users/36737/cypher


---


![Faizy's github stats](https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true)


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=mohd-faizy&layout=compact)](https://github.com/mohd-faizy/github-readme-stats)
