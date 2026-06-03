# The UECGraphScanner Framework
A framework for graph-based scanning and data exploration of unchecked external calls in smart contracts based on bytecode level analysis.

## Table of Contents

- [Installation](#installation)
- [Data Exploration](#data-exploration)
- [Model Training & Evaluation](#model-training--evaluation)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:williamAdson/uecgraphscanner_training_environment.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Exploration
Explore the dataset using the provided Jupyter notebooks.

### Prerequisites
- Install Jupyter Notebook
- Navigate to the root of the project

### Running the Notebooks

1. Change to the notebooks directory:

```bash
cd notebooks
```

2. Start Jupyter Notebook:

```bash
jupyter notebook --no-browser
```

3. Open your browser to http://localhost:8888 and select one of the following notebooks:

- data_analysis.py – Analyzes dataset statistics and distributions
- data_processing.py – Preprocesses and cleans the data
- evaluation.py – Evaluates model performance

## Model Training & Evaluation
Reproduce the experiment by following the steps below.

Data Files:
```bash
import os
import requests

DATASET_PATH = "https://huggingface.co/datasets/Adson59/SmartUecDB/resolve/main/data.tar"
MODEL_PATH = "https://huggingface.co/datasets/Adson59/UECGraphScanner/resolve/main/best_model.pth"

def download_dataset(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
```

Fetch Data:
```bash
url = (DATASET_PATH)
download_dataset(url, 'data')
```

Fetch Model:
```bash
url = (MODEL_PATH)
download_dataset(url, 'checkpoints')
```

Train the model:
```bash
python main.py
``` 

Recover the vocabulary:
```bash
python recover_vocab.py
```

Evaluate the model:
```bash
python evaluate.py
```

## Project Structure
```text
UECGraphScanner/
├── notebooks/           
├── src/                
├── data/               
├── models/              
├── main.py             
├── recover_vocab.py    
├── evaluate.py         
├── requirements.txt   
└── README.md     
```

Data File Structure
```text
data/
├── processed/processed.csv        
├── raw/dataset.csv
├── split/{test.csv,train.csv,val.csv}
```

Model File Structure
```text
models/
├── checkpoints/{best_model.pth,vocab.pth}           
├── saved_models/                
```