# Cluster Leakage Evaluator

**Evaluates the quality of a clustering by examinining the 'leakage' between clusters using the predicted probabilities of a classification model.**

## Overview

Provide a high-level overview of the project, its objectives, and its significance.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Setup](#setup)
4. [Data](#data)
5. [Modeling](#modeling)
6. [Results](#results)
7. [License](#license)

## Project Structure

Explain the structure of your project. Highlight important directories and their purposes.

```
cluster-leakage-evaluator/
    ├── data/                                       <---------- Directory to store data. It is gitignored!
    │    ├── processed/
    │    │
    │    └── raw/
    │
    ├── docs/                                       <---------- Directory of markdown files that will be used to build docs.
    │
    ├── models/
    │
    ├── notebooks/                                  <---------- Test notebooks, following a naming convention
    │                                                           (e.g. `1.0-initial-data-exploration`).
    │
    ├── results/
    │
    ├── secrets/                                    <---------- Directory to store secrets. It is gitignored!
    │
    ├── src/                                        <---------- Directory of main source python files.
    │
    ├── tests/                                      <---------- Directory of unit test.
    │
    ├── .env                                        <---------- Environment variables, use `dotenv` to read them into python
    │                                                           files. It is gitignored!
    │
    ├── .gitignore
    │
    ├── .pre-commit-config.yaml                     <---------- The Git hooks to use at the pre-commit stage.
    │
    ├── README.md
    │
    └── requirements.txt                            <---------- Dependencies requirements to use the repository.
```

## Dependencies

List all dependencies required to run the project, including Python packages and versions. They should be the same as in requirements.txt file.

## Setup

Provide instructions for setting up the project locally. Include steps for installing dependencies and any other necessary setup.
```bash
# Install dependencies
pip install -r requirements.txt
```

## Data

Describe the dataset used in the project. Include information about the source, format, and any data preprocessing steps.

## Modeling

Explain the machine learning models used in the project. Include information about hyperparameters, training process, and any model tuning.

## Results

Present the key findings and results of the project. Include visualizations and any important conclusions drawn.

## License

Equancy All Rights Reserved