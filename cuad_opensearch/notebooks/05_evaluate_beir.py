# %%
import json
import os
import sys

from beir import util
from beir import Dataset
from beir import Evaluate

root_path = Path("/home/ridhi/projects/project1/open_search_tuning")
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
from cuad_opensearch.notebooks.open_search_connect import connect

# Load environment variables
load_dotenv()

# Define the index name
INDEX_NAME = "cuad-index"
REPO_ID = "theatticusproject/cuad"


# Connect to OpenSearch
client = connect()

# Load the CUAD dataset
dataset = Dataset(REPO_ID)  # Update with the actual path to the CUAD dataset

# %%
# Function to evaluate using BEIR
def evaluate_beir():
    # Load the queries and ground truth
    queries = dataset.get_queries()
    ground_truth = dataset.get_ground_truth()

    # Perform search using the hybrid search implementation
    search_results = perform_hybrid_search(queries)

    # Evaluate the search results using BEIR
    evaluator = Evaluate()
    metrics = evaluator.evaluate(search_results, ground_truth)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# Function to perform hybrid search (to be implemented)
def perform_hybrid_search(queries):
    # Placeholder for hybrid search logic
    # This function should return search results based on the queries
    pass
# %%
if __name__ == "__main__":
    evaluate_beir()