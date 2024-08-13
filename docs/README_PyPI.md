# LeakyBlobs

Evaluates the quality of a clustering by examinining the leakage between clusters using the predicted probabilities of a classification model.

---

**NOTE: This is not the full documentation.**
**[Read the docs here.](https://spiffy-bow-8b4.notion.site/LeakyBlobs-b17dd46549f64df4bf617e63d4f3bc01)**

---

## Overview

LeakyBlobs is a python package which provides a sensible alternative to traditional ways of evaluating the quality of a clustering, such as the Elbow Method, Silhouette Score, and Gap Statistic. These methods tend to oversimplify the problem of cluster evaluation by creating a single number which can be difficult to judge for human beings, often resulting in highly subjective choices for clustering hyperparameters such as the number of clusters in algorithms like KMeans. Instead, the LeakyBlobs package is based on the idea that a good clustering is a *predictable* clustering. The package provides tools to train simple classifiers to predict clusters and tools to analyze their probability outputs in order to see the extent to which clusters 'leak' into each other.

## Installation
This package is available through pip using the following command:
```bash
# Install the package
pip install leakyblobs
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

## Usage

Below is a short example of how to use the LeakyBlobs package.

**[Read the full documentation here.](https://spiffy-bow-8b4.notion.site/LeakyBlobs-b17dd46549f64df4bf617e63d4f3bc01)**

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