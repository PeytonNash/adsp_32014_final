# Team Members

Zoe Calianos, Apoorva Gupta, Peyton Nash, Kirthi Rao

# Problem

Personalized Music Recommendation Engine

The goal of this team is to develop a Bayesian solution to current music recommendation systems. We intend to leverage data containing user listening history and artist listening history to build our recommendation engine.

# Previous Work / Approaches

In the past, the more common approaches to building recommendation engines were collaborative filtering & content-based filtering using similarity scores across a user / item vector matrix. Some similarity scores like cosine, jaccard, and pearson are examples. 

We believe a bayesian approach will help us capture low-level latent dimensions in the data while similarity based focusses on surface level proximity between user data. Further, the inclusion of Gaussian priors when thinking about making recommendations allows for more control over regularization by adjusting the variance in priors.

# Data Used

We will be using the publicly available Last.fm dataset, which has over 1 million user/song listening data points. The team will perform necessary data pre-processing to extract the most useful information while keeping computational resource needs in mind.

Link -> [dataset](https://grouplens.org/datasets/hetrec-2011/)

# Data Pre-processing & Cleanng

# Modeling Approaches Considered

the team intends to leverage a Bayesian Probabilistic Matrix Factorization (BPMF) and MCMC approach. The BPMF will be used to learn the histories from a user/song matrix and create our priors from the data. Posterior inference will be carried out using an MCMC approach – we will sample from the posterior distribution to generate personalized recommendations.

Because BPMF operates directly on observed listening data with Bayesian priors, no pre-trained models are necessary for this – the team will build the BPMF using PyMC. The priors and observed song consumption data will provide us the needed information
to make Bayesian inferences.


# Model Used

# Results

# Future Work

Ideally, our training data would include precise timestamps for every listen event, so we could model not just what users like, but when and how their tastes evolve. With time‑stamped histories, we can capture patterns of discovery—e.g. how exposure to one artist in March led to exploring a related genre in June—and ensure that our train/test split respects chronology. In our current “holistic” setup, however, we lose that ordering: we risk recommending a track a user already heard months ago, or—even worse—training on future listens and then “predicting” them in the past. Incorporating temporal dynamics would eliminate these data‑leakage issues, let us model the trajectory of each user’s taste, and ultimately surface genuinely novel, timely recommendations.