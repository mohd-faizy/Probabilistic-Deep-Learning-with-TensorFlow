# **📊 Comprehensive TensorFlow Probability Distribution Reference**

## **🎯 Core Distribution Categories**

### **a) Univariate Distributions**

*Single-variable probability distributions*

| **Continuous** | **Discrete** | **Specialized** | **Heavy-Tailed** |
|----------------|--------------|-----------------|------------------|
| `Normal` | `Bernoulli` | `Beta` | `StudentT` |
| `Uniform` | `Categorical` | `Gamma` | `Cauchy` |
| `Exponential` | `Poisson` | `LogNormal` | `GeneralizedPareto` |
| `Laplace` | `Binomial` | `Weibull` | `HalfCauchy` |
| `Triangular` | `Geometric` | `InverseGamma` | `Horseshoe` |
| `Gumbel` | `NegativeBinomial` | `Chi2` | `GeneralizedNormal` |
| `Logistic` | `Multinomial` | `F` | `VonMises` |

### **b) Multivariate Distributions**

*Joint distributions over multiple variables*

| **Multivariate Normal** | **Simplex & Directional** | **Matrix Distributions** | **Other Joint** |
|------------------------|---------------------------|-------------------------|-----------------|
| `MultivariateNormalDiag` | `Dirichlet` | `MatrixNormalLinearOperator` | `MultivariateStudentT` |
| `MultivariateNormalFullCovariance` | `DirichletMultinomial` | `MatrixTLinearOperator` | `VonMisesFisher` |
| `MultivariateNormalTriL` | `SphericalUniform` | `Wishart` | `MixtureSameFamily` |
| `MultivariateNormalLinearOperator` | `PowerSpherical` | `InverseWishart` | `BatchReshape` |

### **c) Meta-Distributions & Transformations**

*Distributions that modify other distributions*

| **Shape Manipulation** | **Composition** | **Mixture Models** | **Transformations** |
|----------------------|-----------------|-------------------|-------------------|
| `Independent` | `TransformedDistribution` | `MixtureSameFamily` | `QuantizedDistribution` |
| `BatchReshape` | `JointDistribution*` | `Mixture` | `TruncatedDistribution` |
| `ExpandDims` | `Blockwise` | `FiniteDiscrete` | `Masked` |
| `Sample` | `ConditionalTransformedDistribution` | `Empirical` | `DeterministicDistribution` |

***

## **⚙️ Core Operations & Methods**

### **d) Sampling Operations**

*Generate random samples from distributions*

| **Method** | **Output Shape** | **Usage** | **Example** |
|------------|------------------|-----------|-------------|
| `dist.sample()` | `batch_shape + event_shape` | Single sample per batch | `normal.sample()` → `(2,)` if batch_shape=`[2]` |
| `dist.sample(n)` | `[n] + batch_shape + event_shape` | Multiple samples | `normal.sample(100)` → `(100, 2)` |
| `dist.sample([n, m])` | `[n, m] + batch_shape + event_shape` | Multi-dimensional sampling | `normal.sample([10, 5])` → `(10, 5, 2)` |
| `dist.sample(seed=42)` | *Same as above* | Reproducible sampling | For debugging & testing |

### **e) Probability Computations**

*Evaluate likelihood of observed data*

| **Method** | **Input Requirements** | **Output** | **Numerical Stability** |
|------------|----------------------|------------|------------------------|
| `dist.log_prob(x)` | `x` matches event shape | Joint log-probability | ✅ **Preferred** - numerically stable |
| `dist.prob(x)` | `x` matches event shape | Joint probability | ⚠️ Can underflow for small probabilities |
| `dist.cdf(x)` | `x` matches event shape | Cumulative probability | Available for univariate distributions |
| `dist.survival_function(x)` | `x` matches event shape | 1 - CDF | Complementary CDF |

***

## **🔧 Advanced Features & Integration**

### **f) Neural Network Integration**

*Trainable probabilistic layers and parameters*

| **Approach** | **Implementation** | **Use Cases** | **Key Benefits** |
|--------------|-------------------|---------------|------------------|
| **Trainable Parameters** | `tf.Variable` in distribution args | Dynamic loc/scale learning | Full gradient flow |
| **Probabilistic Layers** | `tfp.layers.DistributionLambda` | End-to-end probabilistic models | Keras integration |
| **Custom Layers** | Subclass `tf.keras.layers.Layer` | Domain-specific distributions | Maximum flexibility |
| **Bijector Chains** | `tfd.TransformedDistribution` | Complex transformations | Normalizing flows |

```python
# Example: Trainable Normal Distribution
class TrainableNormal(tf.keras.layers.Layer):
    def __init__(self, event_size):
        super().__init__()
        self.event_size = event_size
        
    def build(self, input_shape):
        self.loc = tf.keras.layers.Dense(self.event_size)
        self.scale = tf.keras.layers.Dense(self.event_size)
        
    def call(self, inputs):
        loc = self.loc(inputs)
        scale = tf.nn.softplus(self.scale(inputs)) + 1e-6
        return tfd.Independent(tfd.Normal(loc, scale), 1)
```

### **g) Shape Broadcasting & Batching**

*Understanding TFP's powerful shape system*

| **Shape Type** | **Purpose** | **Example** | **Sampling Result** |
|----------------|-------------|-------------|-------------------|
| **`sample_shape`** | Independent draws | `[100]` | 100 independent samples |
| **`batch_shape`** | Different parameters | `[3, 2]` | 6 different distributions |
| **`event_shape`** | Multivariate dimensions | `[5]` | 5-dimensional vectors |
| **Final Shape** | Complete tensor | `[100] + [3, 2] + [5]` | `(100, 3, 2, 5)` |

### **h) Loss Functions & Training Patterns**

*Common probabilistic learning objectives*

| **Loss Type** | **Formula** | **Implementation** | **Use Case** |
|---------------|-------------|-------------------|--------------|
| **Negative Log-Likelihood** | `-log p(x\|θ)` | `-dist.log_prob(targets)` | Maximum likelihood estimation |
| **ELBO (Variational)** | `𝔼[log p(x\|z)] - KL[q(z\|x) \|\| p(z)]` | Reconstruction + KL terms | Variational autoencoders |
| **Expectation Loss** | `𝔼[f(x)]` using samples | Monte Carlo estimation | Reinforcement learning |
| **Regularization** | Prior-based penalties | KL divergence to prior | Bayesian neural networks |

***

## **🚀 Professional Usage Patterns**

### **i) Best Practices & Common Patterns**

| **Pattern** | **Implementation** | **Benefits** | **Gotchas** |
|-------------|-------------------|--------------|-------------|
| **Numerical Stability** | Use `tf.nn.softplus()` for positive params | Prevents invalid parameters | Add small epsilon (1e-6) |
| **Memory Efficiency** | Chunked sampling for large N | Manages GPU memory | Balance chunk size vs speed |
| **Reproducibility** | Always set `seed` parameter | Consistent results | Different seeds for train/val |
| **Broadcasting** | Validate shapes before `log_prob` | Avoid silent errors | Use shape debugging utilities |

### **j) Distribution Selection Guide**

| **Data Type** | **Recommended Distributions** | **Key Considerations** |
|---------------|------------------------------|----------------------|
| **Continuous Bounded** | `Beta`, `Uniform`, `TruncatedNormal` | Support constraints |
| **Continuous Unbounded** | `Normal`, `StudentT`, `Laplace` | Tail behavior |
| **Count Data** | `Poisson`, `NegativeBinomial`, `Binomial` | Overdispersion |
| **Categorical** | `Categorical`, `OneHotCategorical` | Number of classes |
| **Probability Vectors** | `Dirichlet`, `DirichletMultinomial` | Simplex constraint |
| **Correlation Matrices** | `LKJ`, `Wishart` | Positive definiteness |

***

## **📈 Advanced Computational Techniques**

### **k) Monte Carlo & Inference Methods**

| **Technique** | **TFP Implementation** | **Complexity** | **Best For** |
|---------------|----------------------|---------------|--------------|
| **Monte Carlo Sampling** | `dist.sample(n_samples)` | ⭐ Basic | Expectation estimation |
| **Importance Sampling** | Custom implementation | ⭐⭐ Intermediate | Rare event simulation |
| **MCMC** | `tfp.mcmc.*` | ⭐⭐⭐ Advanced | Posterior inference |
| **Variational Inference** | `tfp.vi.*` | ⭐⭐⭐ Advanced | Approximate posteriors |
| **Normalizing Flows** | `tfp.bijectors.*` | ⭐⭐⭐⭐ Expert | Complex distributions |