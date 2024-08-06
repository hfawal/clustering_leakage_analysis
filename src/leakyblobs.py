
import numpy as np
import pandas as pd



# Internal method - prepares data. Expects:
# Example ID column (str), target column (int), prediction column (int), probability vector column (vector).
def __prep_data__(predictions: pd.DataFrame, 
                    id_col: str = "ID", 
                    target_col: str = "TARGET",
                    pred_col: str = "PREDICTION",
                    prob_col: str = "PROBABILITY"):
    
    actual_types = predictions.dtypes.to_dict()
    expected_types = {
        id_col: "string",
        target_col: "int",
        pred_col: "int",
        prob_col: "vector"
    }

    for col_name, col_type in expected_types.items():
        if col_name not in actual_types:
            raise ValueError(f"The given dataframe is missing the column '{col_name}'!")
        elif actual_types[col_name] != col_type:
            raise ValueError(f"The column {col_name} has incorrect type {actual_types[col_name]}! Expected: {col_type}")

    predictions = predictions[[id_col, target_col, pred_col, prob_col]].rename(columns={
        id_col: "ID",
        target_col: "TARGET",
        pred_col: "PREDICTION",
        prob_col: "PROBABILITY"
    })

    prob_df = pd.DataFrame(predictions["PROBABILITY"].tolist(), index=predictions.index)
    prob_df.columns = [str(i) for i in prob_df.columns]
    predictions = pd.concat([predictions.drop(columns=['prob_col']), prob_df])

    return predictions



# Class which works with one dataset - saves some repetitive operations by keeping the data.
class ClusterEvaluator:

    # Expects predictions and targets to be integer encoded from 0 to # clusters - 1
    def __init__(self,  
                    predictions: pd.DataFrame, 
                    id_col: str = "ID", 
                    target_col: str = "TARGET",
                    pred_col: str = "PREDICTION",
                    prob_col: str = "PROBABILITY"):

        self.predictions: pd.DataFrame = __prep_data__(predictions, id_col, target_col, pred_col, prob_col)
        self.support: np.ndarray = None
        self.num_clusters: int = predictions["TARGET"].nunique()


    # Get the influence count matrix. For all classes i, j: 
    # Among examples of target class i, how many of them have probability in j above a certain threshold?
    # result[i][j] = influence count of j on target i.
    def get_influence_counts(self, detection_thresh: float = 0.05) -> np.ndarray:

        agg_dict = {str(i): (lambda x: (x >= detection_thresh).sum()) for i in range(self.num_clusters)}
        influence_counts = self.predictions.groupby("TARGET").agg(agg_dict).reset_index()
        influence_counts = influence_counts.sort_values(by="TARGET")

        return influence_counts.drop("TARGET").to_numpy()
    

    # Gets the support of each target, ordered from 0 upwards.
    def get_support(self) -> np.ndarray:

        if self.support != None:
            return self.support
        
        support_df = self.predictions.groupby("TARGET").size().reset_index(name='count').sort_values(by="TARGET")
        self.support = support_df.to_numpy().reshape(-1)

        return self.support
    

    # Influence = influence count normalized by support of class i. Rows do NOT add to 1.
    # Strictly speaking, an influence[i][j] of 5% means that class j affects 5% of the data with target i.
    # Multiple clusters can affect the same data points, which is why rows don't add to 1.
    def get_influence(self, detection_thresh: float = 0.05) -> np.ndarray:

        counts = self.get_influence_counts(detection_thresh)
        support = self.get_support().reshape(-1, 1)

        return counts / support


    # Creates a dictionary from the influence matrix, filtering only the links passing the influence_thresh.
    # print parameter indicates whether to print phrases or not.
    def get_influence_dictionary(self, 
                                 detection_thresh: float = 0.05, 
                                 influence_thresh: float = 0.02,
                                 print = True) -> dict:
        
        # Set diagonal to 0 so that it isn't counted. (Cluster does not "influence" itself).
        influence_matrix = self.get_influence(detection_thresh)
        np.fill_diagonal(influence_matrix, 0)

        influence_counts = self.get_influence_counts(detection_thresh)
        support_arr = self.get_support()

        influence_dict = {}

        for i in range(self.num_clusters):
            for j in range(self.num_clusters):
                if influence_matrix[i][j] >= influence_thresh:
                    influence_dict[(i, j)] = (influence_matrix[i][j], influence_counts[i][j], support_arr[i])

        # Optionally print readable phrases.
        if print:
            influence_dict_sorted = sorted(influence_dict.items(), key=lambda item: item[1][0], reverse=True)
            for item in influence_dict_sorted:
                start, end = item[0]
                prob, count, support = item[1]
                print(f"Cluster {end} influences cluster {end} by {prob:.2%}  ({count} / {support})")   

        return influence_dict


