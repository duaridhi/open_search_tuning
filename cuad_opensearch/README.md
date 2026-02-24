# CUAD OpenSearch Project

This project implements a hybrid search solution using the CUAD (Conversational Question Answering Dataset) dataset with OpenSearch. The goal is to provide efficient document retrieval through a combination of traditional keyword search and semantic search using embeddings.

## Project Structure

```
cuad_opensearch
├── notebooks
│   ├── 01_setup_cuad_index.py        # Sets up the OpenSearch index for the CUAD dataset
│   ├── 02_ingest_cuad_documents.py    # Ingests CUAD documents into OpenSearch
│   ├── 03_hybrid_search.py             # Implements hybrid search functionality
│   ├── 04_evaluate_pytrec_eval.py      # Evaluates search results using PyTREC metrics
│   └── 05_evaluate_beir.py             # Evaluates using the BEIR framework
├── open_search_connect.py               # Connection logic to OpenSearch service
├── indexing_checkpoint.json             # Stores checkpoint information for indexing
├── .env                                  # Environment variables for configuration
└── README.md                             # Project documentation
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine.

2. **Install Dependencies**: 
   Ensure you have Python installed, then install the required packages using:
   ```
   pip install -r requirements.txt
   ```

3. **Environment Variables**: 
   Create a `.env` file in the root directory and add the necessary environment variables, such as API keys and OpenSearch configuration settings.

4. **Set Up OpenSearch Index**: 
   Run the `01_setup_cuad_index.py` notebook to create the OpenSearch index with the appropriate settings and mappings.

5. **Ingest CUAD Documents**: 
   Execute the `02_ingest_cuad_documents.py` notebook to index the CUAD dataset documents into OpenSearch.

6. **Perform Hybrid Search**: 
   Use the `03_hybrid_search.py` notebook to implement and test the hybrid search functionality.

7. **Evaluate Search Results**: 
   Run `04_evaluate_pytrec_eval.py` to evaluate the search results using PyTREC metrics, and `05_evaluate_beir.py` for evaluation using the BEIR framework.

## Usage Guidelines

- Ensure that OpenSearch is running and accessible before executing the notebooks.
- Follow the notebooks in order to set up the index, ingest documents, and perform searches.
- Review the evaluation results to assess the performance of the hybrid search implementation.

## Components

- **OpenSearch Index**: Configured for optimal document storage and retrieval.
- **Document Ingestion**: Efficient bulk indexing of CUAD documents.
- **Hybrid Search**: Combines keyword and semantic search for improved retrieval.
- **Evaluation Metrics**: Implements PyTREC and BEIR evaluations to measure effectiveness.

For further details, refer to the individual notebook files for specific implementations and functionalities.