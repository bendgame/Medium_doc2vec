#Import dependencies
import pandas as pd
import sqlite3
import texthero as hero
from texthero import preprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import tensorflow_hub as hub


#Establish connection to sqlite database
conn = sqlite3.connect("AllPrintings.sqlite")
#load the data into a pandas DataFrame
df = pd.read_sql("select * from cards", conn)

df = pd.read_sql("select distinct name, text, convertedManaCost, power, toughness, keywords from cards where borderColor ='black' and colorIdentity = 'G'", conn)


custom_pipeline = [preprocessing.fillna,
                   #preprocessing.lowercase,
                   preprocessing.remove_whitespace,
                   preprocessing.remove_diacritics
                   #preprocessing.remove_brackets
                  ]
df['clean_text'] = hero.clean(df['text'], custom_pipeline)
df['clean_text'] = [n.replace('{','') for n in df['clean_text']]
df['clean_text'] = [n.replace('}','') for n in df['clean_text']]
df['clean_text'] = [n.replace('(','') for n in df['clean_text']]
df['clean_text'] = [n.replace(')','') for n in df['clean_text']]


df['tfidf'] = (hero.tfidf(df['clean_text'], max_features=3000))


#tokenize and tag the card text
card_docs = [TaggedDocument(doc.split(' '), [i]) 
             for i, doc in enumerate(df.clean_text)]
card_docs


model = Doc2Vec(vector_size=64, min_count=1, epochs = 20)

#instantiate model
model = Doc2Vec(vector_size=64, window=2, min_count=1, workers=8, epochs = 40)

#build vocab
model.build_vocab(card_docs)

#train model
model.train(card_docs, total_examples=model.corpus_count
            , epochs=model.epochs)


#generate vectors
card2vec = [model.infer_vector((df['clean_text'][i].split(' '))) 
            for i in range(0,len(df['clean_text']))]
card2vec

#Create a list of lists
dtv= np.array(card2vec).tolist()

#set list to dataframe column
df['card2vec'] = dtv

df.head(2)

#download the model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

#generate embeddings
embeddings = embed(df['clean_text'])

#create list from np arrays
use= np.array(embeddings).tolist()

#add lists as dataframe column
df['use'] = [v for v in use]

#check dataframe
df.head(2)

df['tsnetfidf'] = hero.tsne(df['tfidf'])
df['tsnec2v'] = hero.tsne(df['card2vec'])
df['tsneuse'] = hero.tsne(df['use'])

#create scatter plot of tfidf
hero.scatterplot(df, col='tsnetfidf', color='convertedManaCost'
                 , title="TF-IDF", hover_data = ['name','text'])
#create scatter plot of doc2vec
hero.scatterplot(df, col='tsnec2v', color='convertedManaCost'
                 , title="Doc2Vec", hover_data = ['name','text'])
#create scatter plot of uni. sent. enc.
hero.scatterplot(df, col='tsneuse', color='convertedManaCost'
                 , title="U.S.E", hover_data = 'name','text'])
