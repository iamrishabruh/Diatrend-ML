# DiaTrend: AI-Driven CGM Trajectory Prediction System
DiaTrend is an AI-driven system designed to predict continuous glucose monitoring (CGM) trajectories, supporting proactive diabetes management. The project addresses key challenges—including temporal variability, sensor noise, and individual differences—by extracting advanced temporal features from CGM data and applying state-of-the-art machine learning models. In doing so, DiaTrend bridges clinical guidelines with modern AI techniques to provide interpretable, real-time predictions.

## Features
- **Advanced Feature Extraction:**

  Extracts both standard time-series and custom clinical features (e.g., glycemic variability and state transition matrices) from DiaTrend CGM data.

- **Diverse Model Architectures:**

  Utilizes cutting-edge machine learning models including:

    - **TabTransformer:** A transformer-based model optimized for tabular data.
    - **Graph Neural Networks (GNN):** Models patient similarities via graph structures.
    - **Attention MLP:** An MLP enhanced with attention mechanisms for focused feature learning.

- **Robust Evaluation Framework:**

  Comprehensive evaluation scripts and Jupyter notebooks facilitate temporal cross-validation, robustness tests (e.g., sensor noise simulation and missing data handling), and clinical significance testing (including comparisons with endocrinologist annotations).

- **Interactive Deployment:**

  Provides a FastAPI-based inference server along with an interactive Streamlit dashboard for real-time predictions, visualization, and hyperparameter tuning.

## Requirements

- **Hardware:**

    - Windows 10/11 (64-bit) PC with at least 8GB RAM.
    - NVIDIA GPU is recommended for model training.

- **Software:**

    - Python 3.9 or later.
    - Libraries: PyTorch, PyTorch Geometric, Streamlit, FastAPI, Jupyter Notebook, and additional packages (see requirements.txt).

- **Data:**

    - DiaTrend CSV files (raw CGM data) along with any supplementary datasets.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/diatrend-system.git
    cd diatrend-system
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv diatrend-env
    diatrend-env\Scripts\activate  # For Windows
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Environment Variables:**

    Set up any required environment variables as needed (refer to config.py for global configuration parameters).

## Usage

**Preprocessing & Feature Extraction**
  
  - Run the preprocessing pipeline to extract temporal features from your CGM data:

      ```bash
      python scripts/preprocess.py
      ```
      *(This script leverages utils/preprocessing.py to compute standard time-series features as well as custom clinical metrics.)*

**Model Training**

  - Train your models using the provided training scripts:

      ```bash
      python scripts/train.py
      ```
      *Configurable parameters (batch size, epochs, learning rate) are available in config.py.*
      *The best-performing model is saved to the checkpoints/ directory.*

**Deployment**

  - API Server:

      Deploy the FastAPI inference server for real-time predictions:

      ```bash
      uvicorn api.api_server:app --reload
      ```
  - Streamlit Dashboard:

      Launch the interactive dashboard:

      ```bash
      streamlit run app.py
      ```
      
      The dashboard allows you to:

        - Upload DiaTrend CSV files.
        - Generate and visualize temporal features.
        - Trigger model training.
        - Query the API for predictions.

## Project Structure

```graphql
diatrend-system/
├── api/                     # FastAPI server for model inference
│   └── api_server.py
├── checkpoints/             # Model checkpoint files
├── config.py                # Global configuration parameters
├── data/                    
│   └── raw/                 # Place your DiaTrend CSV files here
├── evaluation/              # Jupyter notebooks and scripts for evaluation
│   ├── evaluate_model.ipynb
│   └── evaluation.py
├── models/                  # Model definitions
│   ├── tabtransformer.py
│   ├── gnn_model.py
│   └── attention_mlp.py
├── notebooks/               # In-depth analysis and experimental notebooks
│   └── analysis.ipynb
├── preprocessing/           # Preprocessing and feature extraction code
│   └── preprocessing.py
├── scripts/                 # Training and helper scripts
│   └── train.py
├── utils/                   # Visualization and additional helper functions
│   └── visualization.py
├── app.py                   # Streamlit dashboard for interactive experiments
└── project_proposal.md      # Detailed project proposal and methodology
```

## Development Setup

  - **Jupyter Notebooks:**
    Use notebooks in the evaluation/ and notebooks/ directories for exploratory data analysis, hyperparameter tuning, and in-depth evaluation.

  - **Milestone Schedule:**
    Follow the project’s 10-week milestone schedule to guide development from initial setup to final documentation and presentation.

  - **Contributing:**

      - Fork the repository.
      - Create a feature branch.
      - Commit your changes.
      - Push the branch and open a pull request.

## Future Work

  - **Clinical Deployment:**
    Extend the framework for pilot studies in clinical settings to validate model performance and regulatory compliance.

  - **Enhanced Clustering & Personalization:**
    Explore advanced graph-based techniques (e.g., dynamic time warping for patient similarity, multiscale feature extraction) and personalized federated learning to better manage extreme data skew.

  - **Production-Ready Enhancements:**
    Add robust logging, error handling, and extended parameter tuning options for a full-scale production rollout.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, please contact:
[rchouhan.network@gmail.com]

