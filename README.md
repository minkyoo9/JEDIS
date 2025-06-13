# JEDIS
Implementation of **Covering Cracks in Content Moderation: Delexicalized Distant Supervision for Illicit Drug Jargon Detection**, *KDD 2025*

## Overview

JEDIS is a novel framework designed to address the growing challenge of detecting illicit drug jargon in social media content. As online platforms host user-generated content, identifying and moderating discussions involving drug-related jargon has become a critical task. Traditional methods relying on ban lists of terms are easily bypassed through word substitutions and fail to distinguish between euphemistic and benign uses of terms like *pot* or *crack*. 

JEDIS overcomes these limitations by leveraging **context-based analysis** rather than simple keyword detection. It employs a unique combination of **distant supervision** and **delexicalization**, enabling effective training without the need for human-labeled data. This approach is not only robust to new terms and euphemisms but also adaptable to evolving language trends.

---

### - Drug Seed Terms

<img src="https://github.com/user-attachments/assets/e4cec5c7-8cf1-43f6-b4ec-1132b89dd526" width="600">

### - Extracted Top 100 Drug Jargons
<img src="https://github.com/user-attachments/assets/1dcf8be5-8280-40ce-acd6-fe28f477a0be" width="700">

## Requirements

1. **Install Anaconda**: Download and install Anaconda from [this link](https://www.anaconda.com/download).
2. **Create and Activate Python Environment**:
Run the following commands to set up the environment:
```bash
conda env create -f environments.yaml
conda activate jedis
```
---

## Model Evaluation
### Model Preparation
1. **Download the Trained Model**:
   - Download the trained model from [this link](https://github.com/minkyoo9/JEDIS/releases/download/materials/BEST_model.tar.gz).
   - Decompress the file:
     ```bash
     tar -zxvf  BEST_model.tar.gz
     ```
2. **Download the Evaluation Dataset**:
   - Download the data from [this link](https://github.com/minkyoo9/JEDIS/releases/download/materials/data.tar.gz).
   - Decompress the file:
     ```bash
     tar -zxvf  data.tar.gz
     ```

### Run Evaluation
To evaluate the model using the provided (sampled) evaluation dataset, run the following command:

```bash
python Model_train.py
```
> **Note:**
> If you require the full annotated evaluation dataset, please request access for research purposes using the following form: [Google Form](https://forms.gle/3aLdw3SAr1pUf7Z87).


## Model Training
### 1. Dataset Preparation
To prepare the dataset for training, follow these steps:
1. Download the raw data and pretrained model from [this link](https://github.com/minkyoo9/JEDIS/releases/download/materials/data.tar.gz).
2. Use **`Data_processing.ipynb`** to generate the training dataset.
   - This notebook will create datasets in the format:  
     **`Emb_Dataset_ratio{ratio}_max{maxn}_{random_seed}`**.
3. Ensure that the dataset is properly processed and saved for use in training.

### 2. Train the Model
Once the dataset is ready, run the training script using the following command:

```bash
python Model_train.py -t True -r {ratio} -m {maxn} -s {random_seed}
```

- Replace the placeholders with the appropriate values:
  - `{ratio}`: R_nonSTMS.
  - `{maxn}`: R_STMS.
  - `{random_seed}`: The seed for random number generation.
  
> **Note**: Make sure that the training procedure takes several hours, depending on the dataset size and available hardware resources.
