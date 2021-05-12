# Deep Reinforcement Learning (DRL) Method for Restaurant Recommendation Systems

A PyTorch implementation of the DRR framework [1] as it applies to restaurant recommendation.
Please review report.pdf for full explanation.

## Abstract

As cities grow larger, residents are flooded with more restaurant options, increasing indecisiveness in choosing a restaurant to visit or order from. Various models for separating and recommending general items have been developed and are used, but the most common types are limited in various ways. They assume the recommendation process is static as opposed to dynamic between the user and the recommendation system and are tunnel-visioned on the next recommendation without considering future consequences. We examine the DRR framework as it applies to restaurant recommendations. Using the Yelp dataset [2], we achieve a best average reward of 0.65 and best precision of 0.60.

## Repository

This repository contains our code implementation of the Deep Reinforcement Recommendation framework [1] as it applies to restaurant recommendation. We perform data preprocessing (preprocessing.py) and PMF training (data.py) separately. The main algorithm is included in learn.py with the PyTorch models in model.py. The main code is in main.py where most hyperparameters are set. Online and Offline tests can be called from main.py. A quick analysis of learn.py descrives the functions that can be called (namely, there is training, offline, and online testing code, as well as PMF offline testing for benchmark analysis).

The repository also includes two Python notebooks for use in Google Colab. Make sure you specify the path to the necessary python files for running the Python notebooks in Google drive.
One Python notebook (train_PMF_project.ipynb) trains the PMF and exports the trained model parameters for import into the main DRR project.
The other Python notebook (train_drr.ipynb) is the main training and testing code for the DRR project. It is almost a copy of main.py.

We already include our pretrained PMF and preprocessed data for easier execution.

## Contributers

Vele Tosevski
B.A.Sc., M.A.Sc. candidate (vele.tosevski@mail.utoronto.ca)
Faculty of Applied Science & Engineering, University of Toronto

Keegan Poon (keegan.poon@mail.utoronto.ca)
University of Toronto

## References

[1] F.  Liu,  R.  Tang,  X.  Li,  Y.  Ye,  H.  Chen,  H.  Guo,  and  Y.  Zhang,  “Deep  reinforcementlearning based recommendation with explicit user-item interactions modeling,”CoRR, vol.abs/1810.12027, 2018. [Online]. Available: http://arxiv.org/abs/1810.12027

[2] Y. Inc., “Yelp open dataset.” [Online]. Available: https://www.yelp.com/dataset