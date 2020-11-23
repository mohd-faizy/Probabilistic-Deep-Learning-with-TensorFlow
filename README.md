[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![made-with-Markdown](https://img.shields.io/badge/markdown-%23000000.svg?&style=for-the-badge&logo=markdown&logoColor=white)](http://commonmark.org)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/LICENSE)
[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://opensource.com/resources/what-open-source)

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

<p align='center'>
    <a href="https://youtu.be/BrwKURU-wpk" target="_blank"><img src="https://github.com/mohd-faizy/07T_Probabilistic-Deep-Learning-with-TensorFlow/blob/main/Tensorflow_Dev_png/tfp_dev_summit_ytd.jpg" height='350' width='600'></a>
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

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b872c2c7bfaa26b16e8a82beaf72061b48daaf8e">

for `k = 0, 1, 2, ..., n,` where

<img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/d33401621fb832dd2f9783e80a906d562f669008'>

is the binomial coefficient, hence the name of the distribution. The formula can be understood as follows: k successes occur with probability <a href="https://www.codecogs.com/eqnedit.php?latex=p^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^k" title="p^k" /></a> and `n − k` failures occur with probability `(1 − p)^(n − k)`.

### :large_blue_diamond: b) [Poisson Distribution:](https://en.wikipedia.org/wiki/Poisson_distribution)

Poisson distribution is a statistical distribution that shows how many times an event is likely to occur within a specified period of time. It is used for independent events which occur at a constant rate within a given interval of time.

> The Poisson distribution is used to describe the distribution of rare events in a large population. For example, at any particular time, there is a certain probability that a particular cell within a large population of cells will acquire a mutation. Mutation acquisition is a rare event.

#### Characteristics of a Poisson Distribution

The probability that an event occurs in a given time, distance, area, or volume is the same. Each event is independent of all other events. For example, the number of people who arrive in the first hour is independent of the number who arrive in any other hour.

The Poisson distribution is popular for modeling the number of times an event occurs in an interval of time or space.

A discrete random variable `X` is said to have a Poisson distribution with parameter `λ > 0` if for `k = 0, 1, 2, ...,` the probability mass function of X is given by

<img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/c22cb4461e100a6db5f815de1f44b1747f160048'>

where

- `e` is Euler's number `(e = 2.71828...)`
- `k` is the number of occurrences
- `k!` is the factorial of `k.`

The positive real number `λ` is equal to the expected value of `X` and also to its variance.

<img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/2debd3f9adf97c8af4919aa69ed4a7121b47a737'>

The Poisson distribution can be applied to systems with a large number of possible events, each of which is rare. The number of such events that occur during a fixed time interval is, under the right circumstances, a random number with a Poisson distribution.

### :large_blue_diamond: c) [Uniform Distribution:](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)

The distribution in which all outcomes are equally likely. for example: A coin also has a uniform distribution because the probability of getting either heads or tails in a coin toss is the same.

#### The probability density function (PDF) of the continuous uniform distribution is:

<img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/b701524dbfea89ed90316dbc48c5b62954d7411c'>

#### The cumulative distribution function (CDF) is:

<img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/e5c664c7665277eea8f74575f4650fa933f28dcb'>

### :large_blue_diamond: d) [Gaussian Distribution:](https://en.wikipedia.org/wiki/Normal_distribution)

The **Gaussian Distribution**, is a probability distribution that is **Symmetric** about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a **bell curve**.

> Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known.

<img src='https://www.gstatic.com/education/formulas/images_long_sheet/en/normal_distribution.svg'>

_where_

<img src='https://www.gstatic.com/education/formulas/images_long_sheet/en/normal_distribution_normal_distribution_var_1.svg'> = Probability density function

<img src='https://www.gstatic.com/education/formulas/images_long_sheet/en/normal_distribution_normal_distribution_var_2.svg'> = Standard deviation

<img src='https://www.gstatic.com/education/formulas/images_long_sheet/en/normal_distribution_normal_distribution_var_3.svg'> = Mean

Their **Importance** is partly due to the **Central Limit Theorem**. It states that, under some conditions, the average of many samples (observations) of a random variable with finite mean and variance is itself a random variable—whose distribution converges to a normal distribution as the number of samples increases.

### :large_blue_diamond: e) [Exponential Distribution:](https://towardsdatascience.com/what-is-exponential-distribution-7bdd08590e2a)

#### Why did we have to invent Exponential Distribution?

To predict the amount of waiting time until the next event (i.e., success, failure, arrival, etc.).

**\_For example, we want to predict the following:**

- The amount of time until the customer finishes browsing and actually purchases something in your store (success).
- The amount of time until the hardware on AWS EC2 fails (failure).
- The amount of time you need to wait until the bus arrives (arrival).

<img src='https://www.gstatic.com/education/formulas/images_long_sheet/en/exponential_distribution.svg'>

_where:_

- <img src='https://www.gstatic.com/education/formulas/images_long_sheet/en/exponential_distribution_exponential_distribution_var_1.svg'> = Probability Density Function

- `λ` = Rate parameter
- `x` = Random variable

---

# :zero::one: **The TensorFlow Probability library**

---

## :black_circle: **a) Univariate distributions**

> These distributions have an `empty_event` shape, indicating that they are distributions for a **single random variable**.

**Distribution objects** are vital building blocks to build **Probabilistic deep learning Models** as these objects capture the essential operations on probability distributions that we're going to need to build these models.

### :heavy_check_mark: **Defining our first univariate distribution object**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions    # Shortcut to the distributions

'''
Standard normal distribution with 0 mean and standard
deviation equal to 1
'''
normal = tfd.Normal(loc=0., scale=1.)
print(normal)
```

```
# output
tfp.distributions.Normal("Normal", batch_shape=[], event_shape=[], dtype=float32)
```

- `loc` and `scale` : These two **keyword arguments** are required when you instantiate a normal distribution.
- `event_shape=[]` is what captures the dimensionality of the **random variable** itself. Since this distribution is of a single random variable, the event shape is empty, just in the same way that a scalar tensor has an empty shape.

### :heavy_check_mark: **`Sampling()` from the distribution:**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

normal = tfd.Normal(loc=0., scale=1.)
normal.sample()
```

```
# Output
<tf.Tensor: shape=(), dtype=float32, numpy=1.7527679>
```

- One of the key methods for any distribution object is the `sample()` method, which we can use to sample from the distribution.

- If I call the `sample()` method with no arguments, it will return a single sample from this distribution i.e It's a tensor object and has an empty or scalar shape.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

normal = tfd.Normal(loc=0., scale=1.)
normal.sample(3) # We can also draw multiple independent samples from the distribution
```

```
# Output
<tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 1.4466898 ,  0.7341992 , -0.91509706], dtype=float32)>

```

Now it returns a tensor of length three with samples from the standard normal distribution.

### :heavy_check_mark: `prob()` **Method**

This funtion evaluates the **Probability Density Function** at the given input.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

normal = tfd.Normal(loc=0., scale=1.)
normal.prob(0.5) # Evaluating a standard normal PDF at the point 0.5
```

```
# Output
<tf.Tensor: shape=(), dtype=float32, numpy=0.35206532>
```

### :heavy_check_mark: `log_prob()` **Method**

This Method computes the **log probability** at the given input.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

normal = tfd.Normal(loc=0., scale=1.)
normal.log_prob(0.5)
```

The Output obtained is natural logarithm of the previous tense value we obtained
from the prob method.

```
# Output
<tf.Tensor: shape=(), dtype=float32, numpy=-1.0439385>

```

### :heavy_check_mark: **Discrete Univariate distribution object(Bernoulli distribution)**

In the code below the Bernoulli distribution has one parameter, which is the probability that the random variable takes the value 1.Here we're setting this probability to 0.7 using the probs keyword argument

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

bernoulli = tfd.Bernoulli(probs=0.7)
print(bernoulli)
```

```
# Output
tfp.distributions.Bernoulli("Bernoulli", batch_shape=[], event_shape=[], dtype=int32)
```

- Since this is also a univariate distribution, the `event_shape` is empty.

- We can instead instantiate a Bernoulli distribution using the `logits` keyword argument. This might be more convenient depending on how we're using the distribution.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

bernoulli = tfd.Bernoulli(logits=0.847)
print(bernoulli)
```

```
# Output
tfp.distributions.Bernoulli("Bernoulli", batch_shape=[], event_shape=[], dtype=int32)
```

- The relation to the probability value is that the probability is equal to the value of the **sigmoid function** applied to the `logits`.

- The logit's value we can see here, approximately gives the _same Probability value of 0.7_ that we had before.

> **Note:** The Bernoulli constructor requires either the prompts or the logits keyword argument to be provided, but not both.

### :heavy_check_mark: **Sampling from the Bernoulli Distribution**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

bernoulli = tfd.Bernoulli(logits=0.847)
bernoulli.sample(3)
```

Here we drawing `3` independent samples from this Bernoulli distribution, which returns the tensor object shown below.

```
# Output
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 1, 1], dtype=int32)>
```

> **Note:** that the type of this tensor is different, it's an int 32 tensor, since a Bernoulli random variable is discrete, and can only take the values 0 or 1.

### :heavy_check_mark: **Computing the Probabilities from the Bernoulli Distribution**

#### **`prob()` Method**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

bernoulli = tfd.Bernoulli(logits=0.847)
bernoulli.prob(1) # Computing the probability of the event one
```

```
# Output
# Probability of the event one equals to 0.69993746
<tf.Tensor: shape=(), dtype=float32, numpy=0.69993746>
```

#### **`log_prob` Method**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

bernoulli = tfd.Bernoulli(logits=0.847)
bernoulli.log_prob(1)
```

```
# Output
<tf.Tensor: shape=(), dtype=float32, numpy=-0.35676432>
```

### :heavy_check_mark: **`batch_shape` Argument**

Creating another Bernoulli object, Passing in, an array of two values for the probs argument.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_bernoulli = tfd.Bernoulli(probs=[0.4, 0.5])
print(batched_bernoulli)
```

```
# Output
# This object contains a batch of two Bernoulli distributions
tfp.distributions.Bernoulli("Bernoulli", batch_shape=[2], event_shape=[], dtype=int32)
```

> One of the powerful features of distribution objects is that a single object can represent a batch of distributions of the same type.

Since **Bernoulli distribution** is a **Univariate Distribution**, so each of these probability values is used to create a _Separate Bernoulli probability distribution_, both of which are contained within this single Bernoulli object.

- The batch shape is a property of the distribution object, which we can access through the batch shape attribute like this `batched_bernoulli.batch_shape` --> `TensorShape([2])`

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_bernoulli = tfd.Bernoulli(probs=[0.4, 0.5])
batched_bernoulli.sample(3)
```

We recall the `sample(3)` method with 3 as the argument, we'll get three independent samples from both of these Bernoulli distributions in the batch, so the resulting tensor will have shape `3 `by `2`. We can also compute the probability given by each distribution in the batch by passing in an array to the prob method.

```
# Output
<tf.Tensor: shape=(3, 2), dtype=int32, numpy=
array([[0, 1],
       [0, 1],
       [1, 0]], dtype=int32)>
```

**We can also compute the probability given by each distribution in the batch by passing in an array to the prob method**

#### `prob()` Method

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_bernoulli = tfd.Bernoulli(probs=[0.4, 0.5])
batched_bernoulli.prob([1, 1])
```

Here we're computing the probability of the event value 1 for each distribution, which as we expect returns 0.4 and 0.5 in a tensor of length 2.

```
# Output
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.4, 0.5], dtype=float32)>
```

#### `log_prob` Method

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_bernoulli = tfd.Bernoulli(probs=[0.4, 0.5])
batched_bernoulli.log_prob([1, 1])
```

```
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([-0.9162907, -0.6931472], dtype=float32)>
```

### :triangular_flag_on_post: **Que:** What is the shape of the Tensor that is returned from the following call to the sample method :question:

```python
import tensorflow_probability as tfp
tfd = tfp.distributions
batched_normal = tfd.Normal(loc=[-0.8, 0., 1.9], scale=[1.25, 0.6, 2.8])
batched_normal.sample(2)
```

**Ans:** `shape=(2, 3)`

## :black_circle: **b) Multivariate distributions**

### :heavy_check_mark: **Multivariate Gaussian Distribution**

Using the `MultivariateNormalDiag` class to create a two-dimensional diagonal Gaussian

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

mv_normal = tfd.MultivariateNormalDiag(loc=[-1., 0.5], scale_diag=[1., 1.5])
print(mv_normal)
```

```
# Output
tfp.distributions.MultivariateNormalDiag("MultivariateNormalDiag", batch_shape=[], event_shape=[2], dtype=float32)

```

**Note:** If we didn't use the `scale_diag` argument, then the **covariance matrix** would be the **identity matrix** by default, that is the standard deviation of one for each component.

> we can access the event shape by using the following code `mv_normal.event_shape` which is `#(2,)`

#### :small_orange_diamond: **Sampling the Multivariate Distribution**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

mv_normal = tfd.MultivariateNormalDiag(loc=[-1., 0.5], scale_diag=[1., 1.5])
mv_normal.sample(3) # This will produce 3 independent samples from the multivariate distribution.
```

> The distribution has an event shape of `2`, which means that each of those `3` samples will be two-dimensional, so the resulting tensor has a shape of `3` by `2`.

```
# Output
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[-0.68523514,  0.97463423],
       [-1.8620067 , -1.7331753 ],
       [-1.1199658 , -1.0719669 ]], dtype=float32)>
```

### :heavy_check_mark: **Batch Normal Distribution**

Creating a batch normal distribution by passing in an array of values for both the `loc` and the `scale` arguments. This distribution will have a `batch_shape` of two and an empty or scalar `event_shape`, since the normal is a **univariate distribution**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_normal = tfd.Normal(loc=[-1., 0.5], scale=[1., 1.5])
print(batched_normal)
```

```
# Output
tfp.distributions.Normal("Normal", batch_shape=[2], event_shape=[], dtype=float32)
```

#### :small_orange_diamond: **Sampling the Batch Normal Distribution**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_normal = tfd.Normal(loc=[-1., 0.5], scale=[1., 1.5])
batched_normal.sample(3)
```

```
# Output
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[-0.9501647 , -1.041851  ],
       [-0.5308881 , -0.77874947],
       [-2.6619897 , -1.2877599 ]], dtype=float32)>
```

### :large_orange_diamond: **Comparing the Multivariate Gaussian Distribution & Batch Normal Distribution**

```
# Output Multivariate Gaussian Distribution
tfp.distributions.MultivariateNormalDiag("MultivariateNormalDiag", batch_shape=[], event_shape=[2], dtype=float32)

# Output Batch Normal Distribution
tfp.distributions.Normal("Normal", batch_shape=[2], event_shape=[], dtype=float32)
```

- `MultivariateNormalDiag` distribution is a distribution of a **2D** random variable, as `event_shape=[2]`.

- The `Normal` distribution below is a batch of two distributions of a single random variable, `batch_shape=[2]` and `event_shape=[]` being empty.

#### Computing the `log_probs` fo r both

:small_blue_diamond: **For Multivariate Distribution:**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

mv_normal = tfd.MultivariateNormalDiag(loc=[-1., 0.5], scale_diag=[1., 1.5])
mv_normal.log_prob([0.2, -1.8])
```

```
# Output
<tf.Tensor: shape=(), dtype=float32, numpy=-4.1388974>
```

when we pass in a length two array to the `log_prob method`, this array represents a single realization of the two-dimensional random variable. Correspondingly, the tensor that is returned contains a single `log_prob` value.

:small_blue_diamond: **For Batch Normal Distribution:**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_normal = tfd.Normal(loc=[-1., 0.5], scale=[1., 1.5])
batched_normal.log_prob([0.2, -1.8])
```

```
# Output
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([-1.6389385, -2.499959 ], dtype=float32)>
```

Here, the input array represents a value for each of the random variables for the two normal distributions in the batch. The `log_ probs` for each of these two realizations are evaluated and return these two values in a length two tensor, as we can see above.

### :heavy_check_mark: **Batch Multivariate Distribution**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_mv_normal = tfd.MultivariateNormalDiag(
    loc=[[-1., 0.5],[2., 0], [-0.5, 1.5]],
    scale_diag=[[1., 1.5], [2., 0.5], [1., 1.]]
    )

print(batched_mv_normal)
```

```
# Output
tfp.distributions.MultivariateNormalDiag(
    "MultivariateNormalDiag",
    batch_shape=[3],
    event_shape=[2],
    dtype=float32
    )
```

- In the above distributions Each argument is taking a `3` by `2` array. The last dimension corresponds to the `event_size`, and remaining dimensions get absorbed into the `batch_shape`.

- That means that this `MultivariateNormalDiag` distribution has an `event_shape=[2]` and `batch_shape=[3]`. In other words, it contains a batch of three multivariate Gaussians, each of which is a distribution over a two-dimensional random variable.

#### :small_orange_diamond: **Sampling above Distribution**

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

batched_mv_normal = tfd.MultivariateNormalDiag(
    loc=[[-1., 0.5],[2., 0], [-0.5, 1.5]],
    scale_diag=[[1., 1.5], [2., 0.5], [1., 1.]]
    )

batched_mv_normal.sample(2)
```

```
# Output
<tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
array([[[-1.7474746 , -0.39185297],
        [ 2.605815  , -0.6507868 ],
        [-0.2742607 ,  1.7156713 ]],

       [[-0.22726142, -0.8659065 ],
        [ 1.665063  ,  0.9733336 ],
        [-0.57607734,  3.7140775 ]]], dtype=float32)>
```

In the Tensor: `shape=(2, 3, 2)`:

- The first `2` here comes from the `sample_size` of two.
- The `3`is the `batch_size`.
- The final `2` is the `event_size` of the distribution

### :triangular_flag_on_post: Suppose we define the following `MultivariateNormalDiag` object:

```python
import tensorflow_probability as tfp
tfd = tfp.distributions
batched_mv_normal = tfd.MultivariateNormalDiag(
    loc=[[0.3, 0.8, 1.1], [2.3, -0.3, -1.]],
    scale_diag=[[1.5, 1., 0.4], [2.5, 1.5, 0.5]])

# Que: What is the shape of the Tensor returned by the following?
batched_mv_normal.log_prob([0., -1., 1.])
```

**Ans:** `shape=(2,)`

```
# Output
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([ -3.9172401, -11.917513 ], dtype=float32)>
```

> This section shows how `sample`, `batch`, and `event_shapes` are used in distribution objects. And By designing distribution objects in this way, the **TensorFlow probability library** can exploit the **Performance gains** from **Vectorizing Computations**

## :black_circle: **c) The Independent distribution**

## :black_circle: **d) Sampling and log probs**

## :black_circle: **e) Trainable distributions**

---

# :zero::two: **Probabilistic layers and Bayesian neural networks**

## Other Resources

- YouTube [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
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
