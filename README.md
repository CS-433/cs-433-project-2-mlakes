# Smiley prediction on Twitter data

This project is the second group assignment for the Machine Learning course (CS-443) at EPFL. 

We apply machine learning methods to Twitter data to predict if a tweet message used to contain a positive or a negative smiley. 
We present four different models: a simple machine learning baseline model; two long-short term memory (LSTM) models using word2vec and GloVe embeddings respectively; a zero-shot learning inspired model; and a BERT model. 

Our main contribution is the incorporation and comparison of state-of-the-art data representations; transformers and classification models; as well as the use of zero-shot learning for data efficiency. 

Our proposed model is the one that uses CT-BERT language model which achieves **0.906** accuracy and **0.905** f1-score in the test set and it was placed at the third position of the respective AIcrowd competition (submission ID: 107963).

# Colab
For a step-by-step guide to run the project, please take a look at this colab:

<p align="left"><a href="https://colab.research.google.com/drive/1cxs7OSn9n3HlGPBSR77QkY5H0WHjV6Vx?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

We strongly advice running the project with GPU and Colab offers free GPUs. 

# Step-by-step guide

* [Getting started](#getting-started)
    * [Install](#install)
    * [Dependencies](#dependencies)
    * [Data](#report)
* [Modeling](#training-the-model)
    * [training ]
* [Project Architecture](#project-architecture)
* [Running the code](#running-the-code)
    * [Running vanilla models](#running-vanilla-models)
    * [Running model selection](#running-model-selection)
    * [Running final model](#running-final-model)


## Getting started

### Install
Clone and enter the repository
```bash
git clone https://paola-md:<>@github.com/CS-433/cs-433-project-2-mlakes MLProject2
cd MLProject2
```

Project dependencies are located in the `requirements.txt` file. \
To install them you should run:
```bash
pip install -r requirements.txt
```

### Data
The raw data can be downloaded form the webpage of the AIcrowd challenge: \
https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files. \
The data should be located in the `data/` directory in csv format.

To do this, move the zip file to the data directory and run
```bash
unzip data/twitter-datasets.zip -d data/

mv data/twitter-datasets/train_neg.txt data/train_neg.txt 
mv data/twitter-datasets/train_pos.txt data/train_pos.txt 
mv data/twitter-datasets/train_neg_full.txt data/train_neg_full.txt 
mv data/twitter-datasets/train_pos_full.txt data/train_pos_full.txt 
mv data/twitter-datasets/test_data.txt data/test_data.txt
```
# Modeling

## Training
To train the model, you can run
```bash
python src/run.py --training
```

To run a particular model, the name of the model can be passed as a parameter
```bash
python src/run.py --training --glove
```

## Testing
To create the predictions, you can run
```bash
python src/run.py --testing
```
## Complete pipeline
If no parameters are passed, bert model is trained and then the predictions on the test data are made. 
```bash
python src/run.py --testing
``



#### Report
Our paper regarding the methodology and the experiments of the proposed model 
is located under the `report/` directory in pdf format. 


#### Dependencies
Project dependencies are located in the `requirements.txt` file. \
To install them you should run:
```bash
pip install -r requirements.txt
```


## Project Architecture
The source code of this project is structured in the following manner. 

```
project
│  README.md
│  requirements.txt
│
├─docs/                        # documentation of the problem
│
├─data/                        # the data directory
├─saved_models/                # directory to save models
├── src
│   ├── data_cleaning.py
│   ├── data_loading.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── models
│   ├── model_selection.py
│   ├── preprocessing.py
│   └── run.py
└── test
   ├── test_data_cleaning.py
   ├── test_embeddings.py
   └── test_preprocessing.py


Missing (what are we going to do with these?
├── build_vocab.sh
├── cooc.py
├── cut_vocab.sh
├── Dockerfile-notebook
├── glove_template.py
├── pickle_vocab.py
├── project2_description.pdf

```

## Running the code

### Running locally
Runs training and inference of the models 
```bash
python src/run.py
```

### Running colab
To run and assess the vanilla models, please run the following command:
<p align="left"><a href="https://colab.research.google.com/github/geofot96/MLProject2/blob/master/notebooks/MLProject2_GAP.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

### Running docker other VM
The project was dockerized. The project can be easily run in any virtual machine without the need to install any dependencies using our docker containers. 

0. Make sure you have docker[https://docs.docker.com/engine/install/ubuntu/] and git installed and running.  

1. Declare global variables
REPO is availabe in Dockerhub: paolamedo/bert_notebook:latest
```
REPO_URL=paolamedo/bert_notebook:latest
BUILD_DIR=/home/paola/Documents/EPFL/BERT/MLProject2 <location of the cloned repo>
```

3. Run docker
```
docker run --rm -it -e GRANT_SUDO=yes \
--user root \
-p 8888:8888 \
--net=host \
-e JUPYTER_TOKEN="easy" \
-v $BUILD_DIR:/home/jovyan/work $REPO_URL
```

4. Run project from terminal 
```bash
python src/run.py
```
