
# Breast Cancer Subtype Classification

This project explores the classification of breast cancer subtypes using gene expression and clinical data. It implements data preprocessing, exploratory data analysis, and various machine learning models including Random Forest, KNN, SVM, Logistic Regression, and advanced techniques like Multi-Modal Neural Networks, Transformer Model Networks and Graph Convolutional Neural Networks.

## Project Structure

### Data Sources
- **Genetic Data:** Gene expression data (`TCGA_BRCA_tpm.tsv`).
- **Clinical Data:** Patient clinical data (`brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv`).

Both datasets are available for download via the following Google Drive link:
[Download Datasets](https://drive.google.com/drive/folders/1jEYK6SMnU3b7sih1l8gCGzB4lygqkUEP?usp=sharing)

### Notebook Contents
1. **Data Cleaning and Exploration**
   - Loading and previewing genetic and clinical datasets.
   - Merging datasets and preprocessing for analysis.
2. **Visualization**
   - Subtype distribution and relationships between genes.
3. **Machine Learning Models**
   - **Random Forest:** Optimized hyperparameters and feature importance analysis.
   - **KNN:** Neighbor-based classification.
   - **Support Vector Machines:** Tuning hyperparameters.
   - **Logistic Regression:** Simple yet effective classification.
4. **Deep Learning**
   - Multi-Modal Neural Network for gene and clinical data fusion.
   - Graph Convolutional Networks for graph-based classification.
   - Graph Transformer Models for advanced graph representations.
5. **Ensemble Learning**
   - Combining models for improved classification performance.

### Key Features
- Preprocessing pipelines for handling high-dimensional gene data.
- Hyperparameter tuning for each model.
- Confusion matrices and metrics (accuracy, F1-score) to evaluate performance.
- Visual insights into feature importance and gene relationships.

### Dependencies
- **Python Libraries:** pandas, scikit-learn, PyTorch, torch_geometric, matplotlib, seaborn, altair, etc.
- Ensure you have a CUDA-enabled GPU for deep learning models (optional but recommended).

### Setup and Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/breast-cancer-classification.git
   cd breast-cancer-classification
   ```
2. Download datasets from the provided Google Drive link and place them in the `data/` directory.
3. Run the notebook:
   ```bash
   jupyter notebook breast-cancer-subtype-classification_org.ipynb
   ```

### Results
The project achieves promising accuracy and F1-scores across multiple models. Random Forest and Multi Modal models exhibit strong performance in this classification task.

### Acknowledgments
This project uses data from the TCGA BRCA study and the Pan-Cancer Atlas.
