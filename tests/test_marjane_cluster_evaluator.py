from pathlib import Path

PATH_LOCAL = Path(Path.cwd())
PATH_RAW = PATH_LOCAL.parent.parent.absolute()
DATA_PATH = f"{PATH_LOCAL}\\data"

from leakyblobs.leakyblobs import ClusterEvaluator
import pandas as pd
import numpy as np


# Get the data.
test_cluster_predictions = pd.read_parquet(f"{DATA_PATH}\\test cluster data marjane\\pip_package_test_data.parquet")

evaluator = ClusterEvaluator(test_cluster_predictions, "EXAMPLE_ID", "TARGET", "PREDICTION", "probability")

# Run every method.

np.set_printoptions(precision=4, suppress=True)

print("get_influence_counts")
print(evaluator.get_influence_counts(detection_thresh=0.05))
print()

print("get_support")
print(evaluator.get_support())
print()

print("get_influence")
print(evaluator.get_influence(detection_thresh=0.05))
print()

print("get_influence_dictionary")
print(evaluator.get_influence_dictionary(detection_thresh=0.05, influence_thresh=0.02, printPhrases=True))
print()

print("get_total_leakage")
print(evaluator.get_total_leakage(detection_thresh=0.05))
print()

print("hypothesis_test_total_leakage")
fig, _, _, _, _ = evaluator.hypothesis_test_total_leakage(significance_level=0.05)
fig.show()
print()

# These two methods save files in the working directory.

print("create_influence_graph")
evaluator.create_influence_graph(detection_thresh=0.05, influence_thresh=0.02, 
                                 filename="marjane_clustering_influence_graph.html")
print()

print("save_xml_report")
evaluator.save_xml_report(detection_thresh=0.05, influence_thresh=0.02, 
                          significance_level=0.05, filename="marjane_clustering_leakage_report.xlsx")
print()
