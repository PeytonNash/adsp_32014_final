## Team GANGgsters

**Zoe Calianos**, **Apoorva Gupta**, **Peyton Nash**, **Kirthi Rao**

---

## Problem Statement

A person’s taste in movies, television, visual art, and music is difficult to pin down. When Amazon Studios’ production slate failed to garner the audience executives hoped for, Jeff Bezos created a list of twelve characteristics shared by all “iconic” shows. However, it is not difficult to find counterexamples for every item. While a formula for good art remains elusive, streaming services of all kinds face a similar issue: how to use their vast stores of data to make personalized recommendations to their users.

We focus on making **music recommendations**. The traditional “record store” approach learns individual users’ listening habits and identifies artists, albums, or songs that are similar. However, this method has several limitations:

- **User preference understanding:** Why does a user like a particular piece of music?  
- **Content analysis complexity:** Computers struggle to “understand” music at a deep level.  
- **Computational cost:** Calculating similarities for millions of users and items can be prohibitive.

Instead, machine learning approaches often use **collaborative filtering**, which identifies recommendations based on patterns in all users’ listening habits.

Our project utilizes **Markov chain Monte Carlo (MCMC)** and **Automatic Differentiation Variational Inference** to perform **Bayesian Probabilistic Matrix Factorization (BPMF)**, allowing us to predict how many times a user is likely to listen to an artist.

---

## Previous Work / Approaches

- **Collaborative filtering** and **content-based filtering** using similarity scores over a user–item matrix have been the most common approaches.  
  - Popular similarity metrics include cosine, Jaccard, and Pearson scores.
- We believe a **Bayesian approach** will:
  1. Capture low-level latent dimensions in the data  
  2. Provide stronger regularization through Gaussian priors

---

## Data Used

We will be using the publicly available **Last.fm dataset**, which contains over **1 million** user–song listening records.  
   
- **Dataset link:**  
  [HetRec 2011 — Last.fm Dataset (GroupLens)](https://grouplens.org/datasets/hetrec-2011/)

---

## Data Preprocessing and Cleaning

The team tried a variety of data preprocessing methods to extract the most informative features while managing computational resources. Team members created user-item matrices. Some models were based on a subsample of 100 artists and 100 users from a matrix. Others were run on a virtual machine through Google Cloud Platform (GCP) and included the full data. Team members tested raw, log-transformed, and standardized data. The team split the data into training and testing sets for modeling.

---

## Modeling Approaches Considered

1. **Bayesian Probabilistic Matrix Factorization (BPMF)**  
   - Learns latent user and item factors from observed listening counts  
   - Places Gaussian / Negative Binomial priors on these factors for regularization
   - Team experimented with:
          Different likelihood functions (Gaussian, Negative Binomial)
          Hyperparameter tuning (sigma values, latent dimension size)
          Hierarchical priors to improve generalization and model structure
   
2. **MCMC for Posterior Inference**
   - MCMC uses No-U-Turn-Sampler (NUTS) for efficient exploration
   - Samples from the posterior distribution of the latent factors  
   - Generates uncertainty-aware, personalized recommendations via posterior predictive distributions
   - Generates samples that reflect the true posterior distribution, but is computationally expensive
   
3. **ADVI for Posterior Inference**
   - Alternative to MCMC that approximates the posterior using optimization
   - Faster and more scalable, good for large datasets
   - Uses a multivariate Gaussian to approximate the posterior

Because BPMF operates directly on observed listening data with Bayesian priors, no pretrained models are necessary. We implemented BPMF using **PyMC**.

---

## Model Used

*(To be completed)*

---

## Results

*(To be completed)*

---

## Future Work

Ideally, our training data would include **precise timestamps** for every listen event so we could model not only what users like, but **how their tastes evolve over time**. With time‑stamped histories, we would be able to:

- **Capture discovery patterns** (e.g., how listening to Artist A in March led to exploring Genre B in June)  
- Ensure our train/test split respects **chronology**, avoiding data leakage  
- Surface genuinely novel, timely recommendations  

In our current “holistic” setup, we lose ordering information, risking:

- Recommending tracks a user already heard months ago  
- Training on future listens and “predicting” them in the past  

Incorporating **temporal dynamics** will eliminate these issues and improve recommendation quality.
