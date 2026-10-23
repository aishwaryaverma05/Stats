
# Statistics for AI/ML & Placements — Quick Revision Notes

Statistics helps us understand data, identify patterns, make predictions, and evaluate Machine Learning models.

## Table of Contents

1. [Descriptive Statistics](#1-descriptive-statistics)
2. [Probability Fundamentals](#2-probability-fundamentals)
3. [Probability Distributions](#3-probability-distributions)
4. [Sampling and Statistical Inference](#4-sampling-and-statistical-inference)
5. [Statistics for Machine Learning](#5-statistics-for-machine-learning)
6. [Essential Python Code](#6-essential-python-code)
7. [Interview Questions](#7-last-minute-interview-questions)

---

# 1. Descriptive Statistics

Descriptive statistics summarizes the main characteristics of a dataset.

## 1.1 Mean, Median and Mode

### Mean

The average of all observations.

**Formula:**

\[
\bar{x}=\frac{\sum x_i}{n}
\]

Example: Data = 10, 20, 30

Mean = (10 + 20 + 30) / 3 = 20

- Use: Understanding average values and imputing missing numerical data.
- Limitation: Sensitive to outliers.

### Median

The middle value after arranging observations in ascending order.

- Odd number of values: Take the middle value.
- Even number of values: Take the average of the two middle values.

Examples:
- 2, 4, 8 → Median = 4
- 2, 4, 8, 10 → Median = (4 + 8) / 2 = 6

Use: Summarizing skewed data and data containing extreme values.

### Mode

The most frequently occurring value.

Example: 2, 3, 3, 4, 5 → Mode = 3

Use: Imputing missing categorical values, such as city or product category.

| Measure | Meaning | Outlier Sensitivity |
|---|---|---|
| Mean | Average | High |
| Median | Middle value | Low |
| Mode | Most frequent value | Generally low |

## 1.2 Range, Variance and Standard Deviation

### Range

Difference between the maximum and minimum values.

\[
Range=X_{\max}-X_{\min}
\]

Example: 5, 10, 20 → Range = 20 - 5 = 15

### Variance

Measures how spread out values are around their mean.

**Population variance:**

\[
\sigma^2=\frac{\sum(x_i-\mu)^2}{N}
\]

**Sample variance:**

\[
s^2=\frac{\sum(x_i-\bar{x})^2}{n-1}
\]

- Population variance divides by N.
- The usual sample variance estimator divides by n - 1 to correct bias when estimating population variance.

### Standard Deviation

Square root of variance.

\[
\sigma=\sqrt{\sigma^2}
\]

- Low standard deviation: Values are generally closer to the mean.
- High standard deviation: Values are more spread out.

Example: Data = 2, 4, 6

Mean = 4

Squared deviations = 4, 0, 4

Population variance = 8 / 3 ≈ 2.67

Population standard deviation = √2.67 ≈ 1.63

ML use: Understanding feature spread and standardizing features.

## 1.3 Percentiles, Quartiles and IQR

### Percentiles

The p-th percentile is a threshold below which approximately p% of observations fall, depending on the calculation convention.

Example: The 90th percentile is a threshold at or below which approximately 90% of observations lie.

### Quartiles

- Q1 = 25th percentile
- Q2 = 50th percentile (median)
- Q3 = 75th percentile

### Interquartile Range (IQR)

Measures the spread of the middle 50% of observations.

\[
IQR=Q3-Q1
\]

Example: Q1 = 20, Q3 = 50

IQR = 50 - 20 = 30

## 1.4 Outliers

Outliers are observations unusually far from the rest of the data.

### IQR Method

\[
Lower=Q1-1.5(IQR)
\]

\[
Upper=Q3+1.5(IQR)
\]

Values outside these bounds are potential outliers.

### Z-Score Method

\[
z=\frac{x-\mu}{\sigma}
\]

Measures how many standard deviations a value is from the mean.

A common rule flags |z| > 3, especially when the distribution is approximately normal.

### Handling Outliers

- Check for data-entry errors.
- Investigate genuine extreme values.
- Consider capping, transformations or robust methods.
- Do not automatically delete every outlier.

## 1.5 Covariance and Correlation

### Covariance

Measures whether two variables tend to move together.

- Positive covariance: Variables tend to move in the same direction.
- Negative covariance: Variables tend to move in opposite directions.
- Near-zero covariance: Little linear co-movement is indicated.

Sample covariance:

\[
Cov(X,Y)=\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{n-1}
\]

Its magnitude depends on the units of measurement.

### Correlation

Measures the strength and direction of a linear relationship.

\[
r=\frac{Cov(X,Y)}{s_Xs_Y}
\]

Range: -1 to +1.

- +1: Perfect positive linear relationship.
- 0: No linear relationship.
- -1: Perfect negative linear relationship.

Important:
- Zero correlation does not rule out a nonlinear relationship.
- Correlation does not imply causation.

ML use: Exploratory Data Analysis (EDA), feature relationships and identifying potential multicollinearity.

---

# 2. Probability Fundamentals

Probability measures how likely an event is to occur.

## 2.1 Basic Probability Rules

For equally likely outcomes:

\[
P(A)=\frac{\text{Favourable outcomes}}{\text{Total outcomes}}
\]

Example: Probability of heads from a fair coin = 1/2.

### Complement Rule

\[
P(A^c)=1-P(A)
\]

### Addition Rule

\[
P(A\cup B)=P(A)+P(B)-P(A\cap B)
\]

For mutually exclusive events:

\[
P(A\cup B)=P(A)+P(B)
\]

### Multiplication Rule

\[
P(A\cap B)=P(A)P(B|A)
\]

For independent events:

\[
P(A\cap B)=P(A)P(B)
\]

Remember:
- Mutually exclusive events cannot occur together.
- Independent events do not change each other's probabilities.
- These are different concepts.

## 2.2 Conditional Probability

Probability of A occurring when B is already known to have occurred.

\[
P(A|B)=\frac{P(A\cap B)}{P(B)}
\]

Provided P(B) > 0.

Use: Calculating probabilities when additional information is available.

## 2.3 Independent vs Dependent Events

**Independent events:** One event does not change the probability of the other.

\[
P(A|B)=P(A)
\]

Example: Two independent coin tosses.

**Dependent events:** One event changes the probability of the other.

Example: Drawing two cards without replacement.

## 2.4 Bayes' Theorem

Updates a probability using new evidence.

\[
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
\]

Where:
- P(A) = Prior probability
- P(B|A) = Likelihood
- P(B) = Evidence probability
- P(A|B) = Posterior probability

### Example: Medical Test

Suppose:
- Disease prevalence = 1%
- Test sensitivity = 90%
- False-positive rate = 5%

\[
P(D|+)=\frac{0.90(0.01)}
{0.90(0.01)+0.05(0.99)}
\]

\[
=\frac{0.009}{0.0585}\approx15.38\%
\]

Even after a positive result, the probability of disease is approximately 15.38% under these assumptions.

ML use: Naive Bayes and probabilistic reasoning.

## 2.5 Permutations and Combinations

### Permutation

Order matters.

\[
{}^nP_r=\frac{n!}{(n-r)!}
\]

Example: Arranging 2 people from 3 gives 3 × 2 = 6 arrangements.

### Combination

Order does not matter.

\[
{}^nC_r=\frac{n!}{r!(n-r)!}
\]

Example: Selecting 2 people from 3 gives 3 combinations.

Interview shortcut:
- Arrangement → Permutation
- Selection → Combination

---

# 3. Probability Distributions

A probability distribution describes possible values of a random variable and their probabilities.

## 3.1 Random Variables

A random variable assigns a numerical value to the outcome of a random process.

Example: Number of heads in three coin tosses.

### Discrete vs Continuous

- Discrete: Countable values, such as the number of customers.
- Continuous: Measured values, such as height or time.

For a continuous random variable, the probability of any exact single value is zero; probabilities are calculated over intervals.

## 3.2 Bernoulli Distribution

Models one trial with two outcomes: success or failure.

\[
P(X=x)=p^x(1-p)^{1-x}
\]

Where x ∈ {0, 1} and p is the probability of success.

\[
E[X]=p
\]

\[
Var(X)=p(1-p)
\]

Example: A prediction is correct (1) or incorrect (0).

## 3.3 Binomial Distribution

Models the number of successes in n independent trials with the same success probability p.

\[
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
\]

\[
E[X]=np
\]

\[
Var(X)=np(1-p)
\]

Conditions:
- Fixed number of trials.
- Independent trials.
- Two possible outcomes per trial.
- Constant probability of success.

Example: Number of heads in 10 fair coin tosses.

## 3.4 Normal Distribution

A continuous, bell-shaped and symmetric distribution.

- Mean = Median = Mode.
- Most observations lie near the mean.

### Empirical Rule

For an approximately normal distribution:

- About 68% lie within 1 standard deviation of the mean.
- About 95% lie within 2 standard deviations.
- About 99.7% lie within 3 standard deviations.

Use: Understanding distributions and statistical methods that assume approximate normality.

## 3.5 Standard Normal Distribution and Z-Score

The standard normal distribution has mean 0 and standard deviation 1.

\[
z=\frac{x-\mu}{\sigma}
\]

Example: Mean = 50, standard deviation = 10, observation = 70.

\[
z=\frac{70-50}{10}=2
\]

The observation lies 2 standard deviations above the mean.

Important: Standardization does not automatically make data normally distributed.

## 3.6 Poisson Distribution

Models the number of events in a fixed interval under suitable assumptions, including a constant average event rate and independent events.

\[
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}
\]

Where:
- λ = Expected number of events in the interval.
- k = Number of observed events.

\[
E[X]=\lambda
\]

\[
Var(X)=\lambda
\]

Example: Number of support calls received in one hour.

## 3.7 Uniform Distribution

For a continuous uniform distribution on [a, b]:

\[
f(x)=\frac{1}{b-a}
\]

\[
E[X]=\frac{a+b}{2}
\]

\[
Var(X)=\frac{(b-a)^2}{12}
\]

Example: A random number sampled uniformly between 0 and 1.

## 3.8 Expected Value and Variance

### Expected Value

The long-run average value predicted by a probability distribution.

\[
E[X]=\sum_x xP(X=x)
\]

Example: For a fair die:

\[
E[X]=\frac{1+2+3+4+5+6}{6}=3.5
\]

The expected value need not be a possible individual outcome.

### Variance

Measures the spread of a random variable around its expected value.

\[
Var(X)=E[(X-E[X])^2]
\]

Equivalent formula:

\[
Var(X)=E[X^2]-(E[X])^2
\]

---

# 4. Sampling and Statistical Inference

Statistical inference uses sample data to draw conclusions about a larger population.

## 4.1 Population vs Sample

- Population: Entire group being studied.
- Sample: Subset selected from the population.
- Parameter: Numerical characteristic of a population.
- Statistic: Numerical characteristic calculated from a sample.

Example: All company customers form the population; 500 surveyed customers form a sample.

## 4.2 Sampling Methods

- Simple random sampling: Each individual has an equal chance of selection.
- Stratified sampling: Divide the population into strata and sample from each.
- Cluster sampling: Divide into clusters and sample selected clusters.
- Systematic sampling: Select every k-th individual after a starting point.
- Convenience sampling: Select easily accessible individuals; may introduce bias.

## 4.3 Sampling Distribution

The distribution of a statistic, such as the sample mean, across repeated samples of the same size.

It is different from the distribution of individual observations.

## 4.4 Central Limit Theorem (CLT)

Under appropriate conditions, the sampling distribution of the sample mean approaches normality as sample size increases.

\[
E[\bar X]=\mu
\]

\[
SE(\bar X)=\frac{\sigma}{\sqrt n}
\]

The CLT does not say that the original data itself becomes normal.

## 4.5 Standard Error

Measures the sampling variability of a statistic.

For a sample mean:

\[
SE=\frac{\sigma}{\sqrt n}
\]

When population standard deviation is unknown:

\[
SE\approx\frac{s}{\sqrt n}
\]

- Standard deviation: Spread of individual observations.
- Standard error: Sampling variability of an estimate.

Increasing sample size generally decreases standard error.

## 4.6 Point Estimates and Confidence Intervals

### Point Estimate

A single estimate of a population parameter.

Example: Sample mean = ₹500, used to estimate average population spending.

### Confidence Interval

A range of plausible values for a population parameter, constructed using a statistical procedure.

For a mean with known population standard deviation, under suitable assumptions:

\[
\bar{x}\pm z_{\alpha/2}\frac{\sigma}{\sqrt n}
\]

For a 95% confidence interval, z ≈ 1.96.

When population standard deviation is unknown, a t-based interval is commonly used:

\[
\bar{x}\pm t_{\alpha/2,n-1}\frac{s}{\sqrt n}
\]

Interpretation: A 95% confidence procedure captures the true parameter in 95% of repeated samples under its assumptions.

## 4.7 Hypothesis Testing

A statistical procedure used to evaluate evidence against a claim.

- Null hypothesis (H0): The default claim, often no difference or no effect.
- Alternative hypothesis (H1): The claim being investigated, such as a difference or effect.

Example:
- H0: Two models have equal population mean accuracy.
- H1: Their population mean accuracies differ.

## 4.8 P-Value and Significance Level

### P-Value

Assuming H0 is true, the probability of observing a test statistic at least as extreme as the one observed.

### Significance Level (α)

A threshold selected before testing, often 0.05.

Decision:
- If p-value ≤ α: Reject H0.
- If p-value > α: Fail to reject H0.

Important:
- A p-value is not the probability that H0 is true.
- Failing to reject H0 does not prove H0 is true.

## 4.9 Type I and Type II Errors

| Error | Meaning |
|---|---|
| Type I | Rejecting a true H0 (false positive) |
| Type II | Failing to reject a false H0 (false negative) |

- Type I error probability is controlled by the significance level under the test assumptions.
- Type II error probability is denoted by β.
- Statistical power = 1 - β.

## 4.10 Choosing a Statistical Test

| Test | Common Use |
|---|---|
| One-sample Z-test | Compare a mean with a reference when population SD is known and assumptions hold |
| Independent two-sample t-test | Compare means of two independent groups |
| Paired t-test | Compare paired measurements, such as before and after |
| Chi-square test of independence | Check association between categorical variables |
| Chi-square goodness-of-fit | Compare categorical counts with expected proportions |

Test selection depends on data type, study design, independence and distributional assumptions.

---

# 5. Statistics for Machine Learning

## 5.1 Correlation vs Causation

- Correlation: Two variables change together.
- Causation: A change in one variable causes a change in another.

Example: Ice cream sales and drowning incidents may both increase in summer. This does not prove that ice cream causes drowning.

ML lesson: Correlation alone does not establish a causal relationship.

## 5.2 Skewness

Describes the asymmetry of a distribution.

- Symmetric: Both sides are approximately balanced.
- Positive/right skew: Long tail on the right; high values often pull the mean upward.
- Negative/left skew: Long tail on the left; low values can pull the mean downward.

In many right-skewed datasets, mean > median, but this is not universal.

Use: EDA, selecting transformations and interpreting features.

## 5.3 Outlier Treatment

Common methods:
- Correct invalid data.
- Remove erroneous observations when justified.
- Cap extreme values.
- Apply suitable transformations.
- Use robust statistics or models.

Important: Fit data-dependent thresholds and transformations on training data only to prevent data leakage.

## 5.4 Normalization vs Standardization

### Min-Max Normalization

Commonly rescales values to [0, 1].

\[
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

Example: Minimum = 10, maximum = 30, value = 20.

\[
x'=\frac{20-10}{30-10}=0.5
\]

### Standardization

Centers data around zero and scales it by standard deviation.

\[
z=\frac{x-\mu}{\sigma}
\]

| Normalization | Standardization |
|---|---|
| Commonly maps values to [0, 1] | Centers around 0 and scales by SD |
| Uses minimum and maximum | Uses mean and standard deviation |
| Sensitive to extreme values | Also sensitive to extreme values |

ML use: Scaling is important for many distance-based and gradient-based models, such as KNN, K-means and SVM. Tree-based models usually do not require feature scaling.

Fit scalers on training data only.

## 5.5 Class Imbalance

Occurs when one class has many more observations than another.

Example: Fraud detection with 99% legitimate and 1% fraudulent transactions.

A model predicting every transaction as legitimate achieves 99% accuracy but detects no fraud.

### Useful Metrics

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion matrix

### Handling Methods

- Class weights
- Random oversampling
- Random undersampling
- SMOTE, when appropriate

Important: Apply resampling only to training data to prevent leakage into evaluation data.

## 5.6 Statistical vs Practical Significance

- Statistical significance: Evidence against a null hypothesis under a statistical test.
- Practical significance: Whether an effect is large enough to matter in real life.

Example: A statistically significant reduction of 0.01 seconds in processing time may have little practical value.

Consider effect size, confidence intervals, costs and business impact.

## 5.7 A/B Testing

Compares two variants using a defined outcome metric.

Example: Comparing conversion rates for two website designs.

### Basic Workflow

1. Define the metric and hypothesis.
2. Randomly assign users to groups.
3. Run the experiment for an appropriate duration.
4. Compare outcomes using a suitable statistical method.
5. Check effect size, confidence interval and statistical significance.
6. Consider practical impact before deciding.

Common mistakes:
- Stopping as soon as a favourable p-value appears.
- Ignoring sample size and statistical power.
- Ignoring multiple comparisons.
- Confusing correlation with causation.
- Ignoring practical impact.

---

# 6. Essential Python Code

```python
import numpy as np
import pandas as pd

data = pd.Series([10, 20, 20, 30, 40])

# Central tendency
print("Mean:", data.mean())
print("Median:", data.median())
print("Mode:", data.mode().tolist())

# Spread
print("Range:", data.max() - data.min())
print("Population variance:", data.var(ddof=0))
print("Sample variance:", data.var(ddof=1))
print("Population SD:", data.std(ddof=0))

# Percentiles and IQR
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)

# IQR outlier bounds
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("Potential outliers:",
      data[(data < lower) | (data > upper)].tolist())

# Correlation
x = pd.Series([1, 2, 3, 4, 5])
y = pd.Series([2, 4, 6, 8, 10])

print("Correlation:", x.corr(y))
```

### Python Reminders

- `mean()` calculates the mean.
- `median()` calculates the median.
- `mode()` may return multiple modes.
- `var(ddof=0)` uses the population convention.
- `var(ddof=1)` calculates the usual sample variance estimator.
- `quantile()` calculates a requested quantile.
- `corr()` calculates Pearson correlation by default.

---

# 7. Last-Minute Interview Questions

### Q1. Mean vs median?

Mean is the average; median is the middle value. Median is less sensitive to extreme values.

### Q2. Variance vs standard deviation?

Variance measures squared deviations; standard deviation is its square root and uses the original units.

### Q3. What is the Central Limit Theorem?

Under suitable conditions, the sampling distribution of the sample mean approaches normality as sample size increases.

### Q4. What is a p-value?

It measures how extreme the observed test statistic is under the null hypothesis.

### Q5. Type I vs Type II error?

Type I is a false positive; Type II is a false negative.

### Q6. Why is accuracy misleading for imbalanced data?

A model can predict only the majority class and still achieve high accuracy.

### Q7. What is Bayes' theorem used for?

Updating probabilities when new evidence becomes available.

### Q8. When is a t-test useful?

For comparing means when the assumptions of the selected t-test are appropriate, particularly when population SD is unknown.

### Q9. When is a chi-square test useful?

For analysing categorical counts and checking association between categorical variables.

### Q10. Why is feature scaling important?

Many distance-based and gradient-based algorithms are affected by feature scales.

### Q11. Does standardization make data normally distributed?

No. It changes location and scale, not distribution shape.

### Q12. Does correlation imply causation?

No. Confounding variables or other factors may explain the relationship.

### Q13. Standard deviation vs standard error?

Standard deviation describes the spread of observations; standard error describes sampling variability of an estimate.

### Q14. Why apply SMOTE only to training data?

Resampling before splitting can leak information and inflate evaluation performance.

### Q15. Statistical vs practical significance?

Statistical significance concerns evidence under a statistical model; practical significance concerns whether the effect matters in practice.

---

# Final Revision Checklist

- [ ] Mean, median, mode, variance and standard deviation
- [ ] Percentiles, quartiles, IQR and outlier detection
- [ ] Covariance and correlation
- [ ] Probability rules and conditional probability
- [ ] Bayes' theorem
- [ ] Bernoulli, binomial, normal and Poisson distributions
- [ ] Expected value and variance
- [ ] Population, sample, CLT and standard error
- [ ] Confidence intervals and hypothesis testing
- [ ] P-values, significance levels and Type I/II errors
- [ ] Z-test, t-test and chi-square test selection
- [ ] Skewness, feature scaling and class imbalance
- [ ] A/B testing and data leakage prevention

## Last-Minute Revision Strategy

1. Understand what each concept means.
2. Revise its formula and variables.
3. Solve one numerical example.
4. Learn one practical ML application.
5. Practise explaining the concept in simple English.

These notes cover essential Statistics foundations for AI/ML and placements. Advanced topics can be studied separately when required.
