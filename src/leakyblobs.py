
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyvis.network import Network
from scipy.stats import norm



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
        self.count = predictions.shape[0]
        self.targets_arr = self.predictions["TARGET"].to_numpy().reshape(-1)
        self.off_prob_arr = self.predictions.drop("ID", "TARGET", "PREDICTION").to_numpy()
        self.off_prob_arr[np.arange(self.count), self.targets_arr] = 0



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
    

    # Create a graph out of the edges that appear in the influence dictionary above.
    def create_influence_graph(self,
                               detection_thresh: float  = 0.05,
                               influence_thresh: float = 0.02,
                               filename: str = "clustering_influence_graph.html"):

        influence_matrix = self.get_influence(detection_thresh)

        # Create a directed graph
        net = Network(notebook=True, directed=True, height="750px", width="100%", bgcolor="#222222", font_color="black")

        # Add nodes with color based on total connections
        for i in range(self.num_clusters):
            net.add_node(i, label=str(i), shape="circle")

        # Function to determine color brightness based on weight
        def get_color(weight):
            weight = weight / 0.05
            base_color = "255,255,255"  # White in RGB
            intensity = max(0, min(255, int(255 * weight)))  # Adjust intensity based on weight
            return f"rgba({base_color},{intensity})"

        # Add edges with weights, adjusting for bidirectional edges to avoid overlap
        for i in range(self.num_clusters):
            for j in range(self.num_clusters):
                if influence_matrix[i][j] >= influence_thresh:
                    color = get_color(influence_matrix[i][j])
                    edge_label = f"{influence_matrix[i][j]:.2%}"
                    if influence_matrix[j][i] >= influence_thresh:  # Check for bidirectional edge
                        # For bidirectional edges, use "smooth" type to avoid overlap
                        net.add_edge(i, j, title=edge_label, label=edge_label, width=3, 
                                     smooth={'type': 'curvedCW', 'roundness': 0.2}, arrowStrikethrough=False, color=color)
                    else:
                        # For unidirectional edges, use straight lines
                        net.add_edge(i, j, title=edge_label, label=edge_label, width=3, arrowStrikethrough=False, color=color)

        # Set options for the graph to make it more interactive
        net.set_options(
        """
        var options = {
        "nodes": {
            "shape": "circle",
            "size": 50,
            "font": {
            "size": 18,
            "face": "arial",
            "strokeWidth": 0,
            "align": "center",
            "color": "black"
            }
        },
        "edges": {
            "arrows": {
            "to": {
                "enabled": true,
                "scaleFactor": 1
            }
            },
            "color": {
            "inherit": false
            },
            "smooth": {
            "enabled": true,
            "type": "dynamic"
            },
            "font": {
            "size": 14,
            "align": "top",
            "strokeWidth": 0,
            "color": "white"
            },
            "width": 2
        },
        "physics": {
            "forceAtlas2Based": {
            "gravitationalConstant": -100,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
            "enabled": true,
            "iterations": 1000,
            "updateInterval": 25,
            "onlyDynamicEdges": false,
            "fit": true
            }
        }
        }
        """
        )

        net.save_graph(filename)


    # Gets the total leakage: the number of examples with off-probability more than the threshold.
    def get_total_leakage(self, detection_thresh: float = 0.05) -> float:

        max_off_prob = np.max(self.off_prob_arr, axis=1)
        num_leaks = np.sum(max_off_prob >= detection_thresh)

        return num_leaks / self.count
    

    # Do hypothesis testing on the total leakage at different detection thresholds and different comparison values
    # for the alternative hypothesis. Generate a plotly table, also returns data behind the table in a tuple.
    def hypothesis_test_total_leakage(self):

        comparison_values = np.linspace(0.05, 0.25, num=9)
        detection_thresholds = np.linspace(0.05, 0.25, num=9)

        xlen = len(detection_thresholds)
        ylen = len(comparison_values)

        total_leakage_stats = np.zeros(xlen)
        for i in range(xlen):
            total_leakage_stats[i] = self.get_total_leakage(detection_thresholds[i])

        # The null hypothesis is "the total leakage is equal to X"
        # The alternative is "the total leakage is more than X"
        # Performing the grid of hypothesis tests:

        significance_level = 0.05

        n = self.count
        p_0 = comparison_values.reshape(-1, 1)
        p_hat = total_leakage_stats.reshape(1, -1)

        numerator = np.repeat(p_hat, ylen, axis=0) - np.repeat(p_0, xlen, axis=1)
        denominator = np.sqrt(p_0 * (1 - p_0) / n)
        denominator = np.repeat(denominator, xlen, axis=1)
        z_stats = numerator / denominator

        p_values = 1 - norm.cdf(z_stats)
        decisions = p_values < significance_level

        # Format decisions in a table.

        cell_labels = np.round(np.repeat(p_hat, ylen, axis=0), decimals=4)
        cell_colors = decisions.astype(int)
        x_axis = np.round(detection_thresholds, decimals=4)
        y_axis = np.round(comparison_values, decimals=4)

        fig = go.Figure(data=go.Heatmap(
            z=cell_colors,
            text=cell_labels,
            texttemplate="%{text}",
            colorscale=[[0, "green"], [1, "red"]],
            showscale=False  # This line removes the colorbar
        ))

        fig.update_layout(
            title="Hypothesis Testing for Cluster Leakage",
            xaxis=dict(title="Detection Threshold", tickvals=list(range(len(x_axis))), ticktext=x_axis),
            yaxis=dict(title="Comparison", tickvals=list(range(len(y_axis))), ticktext=y_axis)
        )

        fig.show()

        return (fig, cell_labels, cell_colors, x_axis, y_axis)