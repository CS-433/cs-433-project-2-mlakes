# Smiley prediction on Twitter data :)

This project is the second group assignment for the Machine Learning course (CS-443) at EPFL. 

We apply machine learning methods to Twitter data to predict if a tweet message used to contain a positive or a negative smiley. 
We present four different models: a simple machine learning baseline model; two long-short term memory (LSTM) models using word2vec and GloVe embeddings respectively; a zero-shot learning inspired model; and a BERT model. 

Our main contribution is the incorporation and comparison of state-of-the-art data representations; transformers and classification models; as well as the use of zero-shot learning for data efficiency. 

Our proposed model is the one that uses [CT-BERT](https://github.com/digitalepidemiologylab/covid-twitter-bert) language model which achieves **0.906** accuracy and **0.905** f1-score in the test set and it was placed at the third position of the respective AIcrowd competition (submission ID: 107963).

Our **pre-trained model** can be found [here](https://drive.google.com/drive/folders/1aLWxJdPFwOyqvNkkc_QyzhBC9ofY9tgS?usp=sharing).

# Colab
For a step-by-step guide to run the project, please take a look at this notebook:

<p align="left"><a href="https://colab.research.google.com/drive/1cxs7OSn9n3HlGPBSR77QkY5H0WHjV6Vx?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

We strongly advice running the project with GPU and Colab offers free GPUs. 

# Step-by-step guide

* [Getting started](#getting-started)
    * [Install](#install)
    * [Data](#report)
* [Modeling](#training-the-model)
    * [Embeddings](#embeddings)
    * [Training](#training)
    * [Testing](#testing)
    * [Complete pipeline](#complete-pipeline)
    * [Running with Docker](#running-with-docker)
* [Project Architecture](#project-architecture)
    * [Report](#report)
    * [Folder structure](#folder-structure)



## Getting started

### Install
Clone and enter the repository
```bash
git clone https://paola-md:<>@github.com/CS-433/cs-433-project-2-mlakes MLProject2
cd MLProject2
```

We recommend installing the dependencies inside a python virtual environment so you don't have any conflicts with other packages installed on the machine. You can use virutalenv, pyenv or condaenv to do that.
```bash
pyenv virtualenv mlproject2
pyenv activate mlproject2
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

## Embeddings

The BiLSTM can be trained with glove and word2vec embeddings. In order to run these models, we need to create the vocabulary or download a pre-trained one.

### Word2vec
Constructs a a vocabulary list of words appearing at least 5 times.
```bash
src/preprocessing_glove/build_vocab.sh
src/preprocessing_glove/cut_vocab.sh
python preprocessing_glove/pickle_vocab.py
```

### GloVe
You must download the pretrained embeddings from [here](https://nlp.stanford.edu/projects/glove/) or using wget:
```bash
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
mv glove.twitter.27B.zip data/embeddings/glove.twitter.27B.zip
unzip data/embeddings/glove.twitter.27B.zip -d data/embeddings
```

## Training
To train the model, you can run
```bash
python src/run.py --pipeline training 
```

To run a particular model, the name of the model can be passed as a parameter
```bash
python src/run.py --pipeline training \
                  --model glove 
```

The following models can be trained:
* tfidf :   TermFrequency-Inverse Document Frequency 
* word2vec :  BiLSTM using word2vec embeddings
* glove : BiLSTM using glove embeddings
* bert :  Bidirectional Encoder Representations from Transformers (CT-BERT)
* zero : Few shot learning model

To learn more, read the report :D

## Testing
To create the predictions, you can run
```bash
python src/run.py --testing
```
## Complete pipeline
If no parameters are passed, bert model is trained and then the predictions on the test data are made. 
```bash
python src/run.py 
```

### Running with Docker
The project can be easily run in any virtual machine without the need to install any dependencies using our docker container. 

0. Make sure you have [docker](https://docs.docker.com/engine/install/ubuntu/) and git installed and running.  

1. Declare global variables
REPO is availabe in Dockerhub: paolamedo/bert_notebook:latest
```
REPO_URL=paolamedo/bert_notebook:latest
BUILD_DIR=/home/paola/Documents/EPFL/MLProject2 <location of the cloned repo>
```
2. Run docker
```
docker run --rm -it -e GRANT_SUDO=yes \
--user root \
-p 8888:8888 \
-e JUPYTER_TOKEN="easy" \
-v $BUILD_DIR:/home/jovyan/work $REPO_URL
```

3. You will now be able to open jupyter notebook and run notebooks/MLProject2_GAP.ipynb:
```
http://localhost:8888/?token=easy
```
or run from the terminal
```bash
python src/run.py 
```


# Project Architecture

### Report
Our paper regarding the methodology and the experiments of the proposed model 
is located under the `report/` directory in pdf format. 


## Folder structure
The source code of this project is structured in the following manner. 

```
project
├── README.md
├── requirements.txt
├── Dockerfile-notebook
├─docs/                        # report and project description
│
├─data/                        # the data directory
│   ├── embeddngs/             # dirctory where embeddings will be stored
│   └── twitter-datasets.zip   # This is where the data should be loaded
├── models/                    # directory where models are saved
├── predictions/               # directory where the predictions are saved
├── notebooks
│   └── MLProject2_GAP.ipynb
├── src
│   ├── models/                # directory with models' code   
│   ├── preprocessing_glove/   # directory with files to preprocess corpus for glove
│   ├── data_cleaning.py
│   ├── data_loading.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── model_selection.py
│   ├── preprocessing.py
│   └── run.py
└── test                       # unit tests
   ├── test_data_cleaning.py
   ├── test_embeddings.py
   └── test_preprocessing.py

```



