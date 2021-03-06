"#!/usr/bin/env python"

"""
Script for generating embeddings from a supplied file and saving them. 
Also for generating tsv files of the vectors and the metadata to enable visualising them.
AUTHOR : Lawrence Adu-Gyamfi
DATE : 03/06/2020

**********************
Update : 08/06/2020
Add additional option to use pretrained model

Update : 11/06/2020
Add command line options and flags
Add functions and code for computing correlations with wordsim353 
"""

import numpy as np
import unicodedata
import re
import os, shutil
from gensim.models import Word2Vec, FastText, fasttext
from gensim.test.utils import datapath
import pandas as pd
from scipy.stats import spearmanr
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Name of datafile to be used for the training")
parser.add_argument("-t", "--test", help="Indicate if current run is for a test or actual training", action="store_true")
parser.add_argument("-v", "--visualize", help="Save TSV files of tensors and metadata for visualization on embedding projector", action="store_true")
parser.add_argument("-s", "--save_model", help="Save learned model", action="store_true")
parser.add_argument("-c", "--corr", help="Compute correlation with wordsim dataset", action="store_true")
parser.add_argument("--init", help="Initialize using a pretrained word embedding model")
parser.add_argument("--epochs", help="Number of epochs to train new embeddings model", type=int)
args = parser.parse_args()

NUMBER_OF_DATASET = 100
DIMENSION = 300
DATA_DIR = "./DATA"
MODELS_DIR = "./MODELS"
ENG_PATH = os.path.join(DATA_DIR, "jw300.en-tw.en")
TWI_PATH = args.data if args.data else os.path.join(DATA_DIR, "jw300.en-tw.tw")
WORDSIM_PATH = os.path.join(DATA_DIR, "wordsim_tw.txt")

######################### RUNNING PARAMETERS ####################
TEST = True if args.test else False
VISUALIZE_FILE = True if args.visualize else False
SAVE_MODEL = True if args.save_model else False
COMPUTE_CORRELATION = True if args.corr else False
USE_PRETRAINED = True if args.init else False
EPOCHS = args.epochs if args.epochs else 1
#################################################################
init_filepath = args.init if args.init else None


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_line(s, language="eng"):
    """
    Perform some cleanup on supplied str based on language.
    
    Parameters
    ----------
    s : str
    language: str
        default is "eng" for english. option is "twi"
    
    Returns
    -------
    str of cleaned sentence
    """
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = s.lower()
    if language == "twi":
        s = re.sub(r'[^a-zA-Z.ƆɔɛƐ!?’]+', r' ', s)
    elif language == "eng":
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

def read_dataset(file_path, number=None, normalize=False, language="eng"):
    """
    Read NUMBER_OF_DATASET lines of data in supplied file_path
    Perform normalization (if normalize=True) based on input language(default:"eng", option:"twi")

    Returns
    -------
    List[list] of processed word tokens for sentences in file_path
    """

    with open(file_path) as file:
        data = file.read()
    data = data.split("\n")
    if number:
        assert number < len(data), "Number of dataset less than required subset"
        data = data[:number]
    if normalize:
        data = [normalize_line(line, language=language).split() for line in data]
    return data

def get_embedding(data, typeFunc=Word2Vec, size=100, window=5, min_count=5, sg=0, save=False, negative=10):
    """
    Generate embeddings for input data. Currently works with either word2vec or Fasttext from gensim

    Parameters
    ----------
    data : list[list]
        preprocessed word tokens
    typeFunc : gensim.model name
        Either Word2Vec or FastText (without "") (default is Word2Vec)
    size : int
        Dimension of embeddings to be generated (default=100)
    window : int
        size of window to be considered for word embeddings (default=5)
    min_count : int
        
    sg : int (0,1)
    
    save : bool
        if True, save generated embeddings in current working directory of script

    Returns
    -------
    Embeddings of type gensim.model
    """
    embeddings  = typeFunc(data,size=size, window=window, min_count=min_count, workers=4, sg=sg, negative=negative)
    if save:
        embeddings.save(f"{MODELS_DIR}/{typeFunc.__name__}_embedding.mod")
    return embeddings



def get_embedding_train(data, model, typeFunc=fasttext, epochs = 5, size=100, window=5, min_count=5, sg=0, save=False, negative=10):
    """
    Generate embeddings for input data using a pretrained model. Currently works with either word2vec or Fasttext from gensim

    """
    # load the pretrained model
    print("Loading pretrained model ..\n")
    embeddings = typeFunc.load_facebook_model(datapath(os.path.abspath(model)))
    # build the vocab
    embeddings.build_vocab(sentences=data, update=True)
   # embeddings  = typeFunc(size=size, window=window, min_count=min_count, workers=4, sg=sg)
    print("Training embeddings ...\n")
    embeddings.train(sentences=data, epochs=epochs,word_count=0, total_examples=embeddings.corpus_count, window=window, min_count=min_count, sg=sg, total_words=embeddings.corpus_total_words, negative=negative)
    if save:
        embeddings.save(f"./{typeFunc.__name__}_embedding.mod")
    return embeddings



def prepare_for_visualization(model, model_path=None, save_dir="."):
    """
    Generates tsv formats of metadata and tensors/vectors for embeddings. 
    Useful for tensorflow embeddings projector.

    Parameters
    ----------
    model : gensim model type
        embeddings created using either word2vec or fasttext
    model_path : path, optional
        Path to a saved embeddings file (default is None)
    save_dir : path
        Path to directory to save created tsv files. (default is current working directory of script)

    Returns
    -------
    A tuple of tensors and metadata of embeddings
    """
    if model_path:   # to do -> check correctness of path
        model = gensim.models.KeyedVectors.load_word2vec_format(f"{model_path}", binary=False, encoding="utf-16")
    with open(f"{save_dir}/embedding_tensors.tsv", 'w+') as tensors:
        with open(f"{save_dir}/embedding_metadata.dat", 'w+') as metadata:
            for word in model.wv.index2word:
                #encoded=word.encode('utf-8')
                encoded = word
                metadata.write(encoded + '\n')
                vector_row = '\t'.join(map(str, model[word]))
                tensors.write(vector_row + '\n')
    return tensors, metadata

def get_similarity(word1, word2, model):
    """
    Return cosine similarity between word1 and word2 using the supplied model

    Parameters
    ----------
    word1 : str
    word2 : str
    model : gensim model type of learned word embeddings

    """
    return model.wv.similarity(word1, word2)



if __name__ == "__main__":
    import datetime, time
    
    if TEST:
        print("This is just a test run!\n")

    GLOBAL_START = time.time()  # to track time for the entire run
    DATE = datetime.datetime.now()
    START_DATE = DATE.strftime("%d/%m/%Y %H:%M:%S")
    
    # create logger file to log details of run
    logger = open(f"log.txt_{START_DATE.split()[1]}", "w")
    
    logger.write(f"Run started on : {START_DATE}\n")
    logger.write(" ================================================================= \n")
    
    # Read twi data from supplied path to TWI file and preprocess
    print("Reading and processing dataset ...\n")
    start =time.time()
    number = NUMBER_OF_DATASET if TEST else None
    twi_data = read_dataset(TWI_PATH, number = number, normalize=True, language="twi")
    tot_time = time.time() - start
    logger.write(f"Time to complete reading file : {tot_time:.2f}\n")
    
    # create embeddings from preprocessed twi data
    print("Creating Embeddings ...\n")
    start =time.time()
    dimension = 50 if TEST else DIMENSION
    if USE_PRETRAINED:
        embeddings = get_embedding_train(twi_data, init_filepath, sg=1, negative=10, size=dimension, epochs=EPOCHS)
    else:
        embeddings = get_embedding(twi_data, FastText, size = dimension, sg=1, negative=10,  save=SAVE_MODEL)
    tot_time = time.time() - start
    logger.write(f"Time to complete creating embeddings file : {tot_time:.2f}\n")
    model_details = str(embeddings)
    logger.write(f"Model Details : {model_details}\n")
    print(f"Model Details : {model_details}")
    
    # generate tsv files for the tensors and the meta to be used for visualization
    if VISUALIZE_FILE:
        print("Generating TSV files for visualization ...\n")
        start = time.time()
        prepare_for_visualization(embeddings, save_dir=MODELS_DIR)
        tot_time = time.time() - start
        logger.write(f"Time to complete creating TSV  file : {tot_time:.2f}\n")
    
    #compute correlations with wordsim data
    if COMPUTE_CORRELATION:
        print("Computing Correlation")
        word_sim = pd.read_csv(WORDSIM_PATH, header=None,)
        word_sim.columns = ["word1", "word2", "relatedness"]
        similarities = [get_similarity(*word_sim[["word1","word2"]].values[i], embeddings) for i in range(len(word_sim))]
        word_sim['similarities'] = similarities
        corr =spearmanr(word_sim[["relatedness", "similarities"]])
        logger.write("==================================================================\n")
        logger.write(str(corr))
        logger.write("\n==================================================================\n")
        wordsim_update = f"wordsim_{START_DATE.split()[-1]}.csv"
        word_sim.to_csv(wordsim_update, index=False)
        print(f"Correlation is : {corr}\n")


    logger.write(f"Run completed on : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

    logger.close()

    print("Completed Run successfully!\n")
