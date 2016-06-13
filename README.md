# Code for the Expedia competition on Kaggle

A simple machine learning algorithm for the Expedia competition on Kaggle.

It is designed to run with low memory (<4GB).

## Content:

- elaborate_features is a notebook for some prelimary analisys of the data, and generation of new features.
- counter_and_leak analizes creates counters of the most common hotel_cluster with respect to the other features, then weights them to generate a predictor. It then generates a leaked solution (due to a leak in the data, since in some cases few features determine the true prediction) and updates the generated one accordingly.
- the other notebooks generate a .py file to run outside the notebook to improve performance.
