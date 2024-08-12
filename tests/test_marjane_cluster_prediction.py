from pathlib import Path

PATH_LOCAL = Path(Path.cwd())
PATH_RAW = PATH_LOCAL.parent.parent.absolute()
DATA_PATH = f"{PATH_LOCAL}\\data"

from leakyblobs.leakyblobs import ClusterEvaluator, ClusterPredictor
import pandas as pd



# Get the data.
training_set = pd.read_parquet(f"{DATA_PATH}\\test cluster data marjane\\pip_package_full_cluster_data.parquet")
training_set.TARGET = training_set.TARGET.astype("int32")
training_set.info()


# Run methods for prediction modeling.
predictor = ClusterPredictor(training_set,
                             "GSM_ACTUEL",
                             "TARGET")

predictor.evaluate_model()

output = predictor.get_test_predictions()
output.info()


# Try passing it directly to the cluster evaluator
evaluator = ClusterEvaluator(output)
evaluator.save_leakage_graph(filename="blob_graph.html")
evaluator.save_xml_report(filename="blob_report.xlsx")