# Cluster Leakage Evaluator

Evaluates the quality of a clustering by examinining the 'leakage' between clusters using the predicted probabilities of a classification model.

---

**NOTE: This README does not contain the full documentation.**
**[Read the docs here.]() (TODO: ADD LINK TO DOCS)**

---

## Overview

This project is a PyPI package which provides a sensible alternative to traditional ways of evaluating the quality of a clustering, such as the "Elbow Method," Silhouette Score, and Gap Statistic. These methods oversimplify the problem of cluster evaluation by creating a single number which can be difficult to judge for human beings, often resulting in highly subjective choices for clustering hyperparameters such as the number of clusters in algorithms like KMeans. Instead, the `leakyblobs` package contained in this project is based on the idea that a good clustering is a *predictable* clustering. The package provides tools to train simple classifiers to predict clusters and tools to analyze their probability outputs in order to see the extent to which clusters 'leak' into each other.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Setup](#setup)
4. [Data](#data)
5. [Usage](#usage)
7. [License](#license)

## Project Structure

```
cluster-leakage-evaluator/
    ├── data/                                       <---------- Directory to store data for testing.
    │
    ├── docs/                                       <---------- Directory of markdown files that will be used to build docs.
    │
    ├── leakyblobs/                                 <---------- Directory of main module for the package.
    │
    ├── tests/                                      <---------- Directory for tests.
    │
    ├── .gitignore
    │
    ├── .pre-commit-config.yaml                     <---------- The Git hooks to use at the pre-commit stage.
    │
    ├── README.md                                   <---------- The file you're reading right now.
    │
    └── requirements.txt                            <---------- Dependencies requirements to use the repository.
```

## Dependencies

```
numpy>=1.26.1
pandas>=2.0.0
openpyxl>=3.1.5
pyvis>=0.3.2
plotly>=5.20.0
scipy>=1.14.0
openpyxl>=3.1.5
setuptools>=72.1.0
scikit-learn>=1.5.1
```

## Setup

The package that this project offers has already been uploaded to [PyPI](https://pypi.org/). To use it, simply
```bash
# Install the package
pip install leakyblobs
```

To install the package's necessary dependencies for editing the project code, use:
```bash
# Install dependencies
pip install -r requirements.txt
```
If you have any issues with importing in the `tests` folder, `pip install -e .` should resolve them.

To re-upload the package to [PyPI](https://pypi.org/), you will also need to `pip install twine`. A good tutorial on how to export packages can be found [here](https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/). For authentification, after you create a PyPI account and an API token therein, use a [`.pypirc`](https://packaging.python.org/en/latest/specifications/pypirc/) file.


## Data

For testing purposes, a dataset containing features used to cluster customers of [Marjane](https://www.marjane.ma/), a client of [Equancy | Groupe EDG](https://www.equancy.fr/fr/), was useful. The `data` folder is not included in the repository to protect their privacy.
(For readers at Equancy | Groupe EDG, this data can be found in the sandbox cloud storage.) 

## Usage

Below is a short example of how to use the `leakyblobs` package.

**[Read the full documentation here.]() (TODO: ADD LINK TO DOCS)**

```python

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from leakyblobs import ClusterPredictor, ClusterEvaluator

# Load iris data set as pandas DF, and concatenate target with features.
iris = load_iris()
data = pd.DataFrame(
    np.concatenate((iris.data, np.array([iris.target]).T), axis=1), 
    columns=iris.feature_names + ['target']
)
data = data.reset_index()
data["index"] = data["index"].astype("str")
data["target"] = data["target"].astype("int32")

# Use the leakyblobs package to train a cluster classification model.
predictor = ClusterPredictor(data, 
                             id_col="index", 
                             target_col="target",
                             nonlinear_boundary=True)

# Get the predictions and probability outputs on the test set.
test_predictions = predictor.get_test_predictions()

# Use the leakyblobs package to evaluate the leakage of a clustering
# given a cluster classification model's predictions and probability outputs.
evaluator = ClusterEvaluator(test_predictions)

# Save visualization in working directory.
evaluator.save_leakage_graph(detection_thresh=0.05,
                             leakage_thresh=0.02,
                             filename="blob_graph.html")

# Save report with leakage metrics in working directory.
evaluator.save_leakage_report(detection_thresh=0.05,
                              leakage_thresh=0.02,
                              significance_level=0.05,
                              filename="blob_report.xlsx")
```

## License

Equancy All Rights Reserved