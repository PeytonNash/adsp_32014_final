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

Our project utilizes **Markov chain Monte Carlo (MCMC)** to perform **Bayesian Probabilistic Matrix Factorization (BPMF)**, allowing us to predict how many times a user will listen to an artist.

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

- The team will perform necessary data preprocessing to extract the most informative features while managing computational resources.  
- **Dataset link:**  
  [HetRec 2011 — Last.fm Dataset (GroupLens)](https://grouplens.org/datasets/hetrec-2011/)

---

## Data Preprocessing and Cleaning

*(To be completed)*

---

## Modeling Approaches Considered

1. **Bayesian Probabilistic Matrix Factorization (BPMF)**  
   - Learn latent user and item factors from observed listening counts  
   - Place Gaussian / Negative Binomial priors on these factors for regularization  
2. **MCMC / ADVI for Posterior Inference**  
   - Sample from the posterior distribution of the latent factors  
   - Generate personalized recommendations based on posterior predictive distributions

Because BPMF operates directly on observed listening data with Bayesian priors, no pretrained models are necessary. We will implement BPMF using **PyMC**.

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
