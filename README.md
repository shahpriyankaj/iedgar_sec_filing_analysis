## IEDGAR SEC Filing Public Corpus Dataset Analysis using K-means Clustering and QnA using Semantic Search and LLM 

### Project Structure
1. <b>edgar_kmeans.ipynb</b>: Task 1: Engineering
2. <b>edgar_genai.ipynb</b>: Task 2: GenAI
3. <b>plots/</b>: Plots from Task 1

### Usage/Reproducability
1. Clone the Repository
```bash
git clone https://github.com/shahpriyankaj/iedgar_sec_filing_analysis.git
cd iedgar_sec_filing_analysis
```
2. Create a Virtual Environment
```bash
python3 -m venv venv
venv\Scripts\activate #On Unix, source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
'''
To download the LLM Model, perform the following steps and copy the model to working directory OR from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf -> download the model to working directory
'''
pip3 install huggingface-hub>=0.17.1
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```
4. Run the provided python notebooks
5. Limitations on Windows to run PySpark:
    Since I have Windows machine, I face multiple issues working on PySpark, hence I switched to Google Collab. Both notebooks can be uploaded to Google Collab and execute each cell to install PySpark as well as mount google drive to download embedding and LLama models. 
6. Load the dataset from https://huggingface.co/datasets/eloukas/edgar-corpus. The code is already present in the notebooks.

### Task #1 - Engineering
This task creates a solution that allows the end user to understand the documents in a two dimensional space to understand different clusters and identify outliers.

#### Dataset Filtering Criteria
- Year: 2020
- Sections: All
- Company: 10 distinct companies

#### Assumptions
- Transforming all sections of single filing into rows enables semantic analysis and clustering, facilitating 2D visualization and outlier detection based on meaningful text content, rather than having all sections combined.
- Paragraph-based chunking is used for semantic analysis.
- The system generated Embeddings using all-MiniLM-L6-v2 model. It is Chosen for balance of speed & semantic quality
- Scaling is performed using StandardScaler for PCA and K-means modeling
- PCA is used to reduce noise and speed up clustering, with n_components chosen based on explained variance.
- KMeans with k=5 is used for clustering.
- Outliers are defined as data points with a distance from their cluster centroid greater than or equal to the mean plus two standard deviations.

#### Future Enhancements
- Experiment with different chunking methods (e.g., sentence-based, fixed-length, semantic chunking, markdown, etc.).
- Explore alternative outlier detection techniques (e.g., z-score, IQR based on data distribution).
- Optimize PCA using the elbow method or a more robust approach for n_components selection.
- Implement automatic K selection for KMeans using the elbow method.
- Consider alternative dimensionality reduction methods for visualization.
- Explore treating entire sections as single documents and Compare results with separate rows for each section.
- Explore alternative visualizations beyond scatter plots.

#### Output
The code generates three scatter plots:
- Embeddings by assigned clusters.
- Embeddings by outlier flag.
- Embeddings by section number.


### Task 2: GenAI: Edgar Corpus Q&A System
This task demonstrates a question-answering using the Edgar Corpus dataset from Huggingface and the Llama 2 LLM.

#### Dataset Filtering Criteria
- Year: 2018-2020
- Sections: All
- Company: First company chosen with most filing documents (i.e. cik: 1001907)

#### Assumptions
- The system assumes all questions are within the context of financial documents from the Edgar Corpus.
- The system uses RecursiveCharacterTextSplitter for chunking, assuming it's suitable for financial text.
- The system generated Embeddings using all-MiniLM-L6-v2 model. It is Chosen for balance of speed & semantic quality
- The system uses Llama-2-7B-Chat model and Prompt engineering with System message, context and query, assuming this task will eventually be used as QnA chatbot.
- The system primarily focuses on extracting information from the year mentioned in the query.
- A default year (2020) is used if the LLM fails to extract the year from the query.
- For validation, golden examples are manually created with Section 1 of the year 2020 for company 1001907 using chatgpt and stored in a Pandas DataFrame. Ideally, business users would provide and store these examples in persistent storage.
- Evaluation uses ROUGE (Rouge-1, Rouge-2, Rouge-3, Rouge-L) scores for comparing LLM-generated answers with golden examples. This will demonstrate that the LLM can retrieve the correct chunks from the embedding object for the correct year (year 2020 in this case)

#### Future Enhancements
- Experiment with different chunking methods (e.g., sentence-based, fixed-length, semantic chunking, markdown, etc.)
- Enhance year extraction to handle edge cases and use regex as a backup.
- Store golden examples in persistent storage for easier management and evaluation.
- Evaluate LLM performance using additional metrics like Bleu and BERTScore.
- Integrate the system with a user interface for interactive querying.


