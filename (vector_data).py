# print(vector_data)

import sqlite3

from gensim.models import Word2Vec

import pandas as pd

 

# Load the .db file into a Pandas DataFrame

conn = sqlite3.connect(r'c:\Users\a889563\Downloads\chinook.db')

df = pd.read_sql('SELECT * FROM albums;', conn)

 

# Extract the text data from the DataFrame

text_data = df['Title'].astype(str)  # Ensure text data is string type

 

# Split the text into list of words (assuming words are already preprocessed or separated)

tokenized_text = [text.split() for text in text_data]

 

# Train Word2Vec model

word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

 

# Save the Word2Vec model

print(word2vec_model)

word2vec_model.save(r'c:\Users\a889563\Downloads\word2vec_model.bin')

 

# Access vectors for individual words

# For example, to get the vector for the word "example":

# word_vector = word2vec_model.wv['example']

 

conn.close()  # Close the database connection

from gensim.models import Word2Vec

 

# Load the pre-trained Word2Vec model from the binary file

model = Word2Vec.load(r'c:\Users\a889563\Downloads\word2vec_model.bin')

 

vocabulary = list(model.wv.index_to_key)

 

# Iterate over the vocabulary and print each word and its corresponding vector

for word in vocabulary:

    vector = model.wv[word]

    print(f'Word: {word}, Vector: {vector}')