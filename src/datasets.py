'''
Make scripts to download and process datasets. We should use lyricsgenius or Genius API to get lyrics.
The datasets should be in the following format:
    Columns: ['decade', 'genre', 'song_title', 'artist_gender',  'lyrics']
    - decade: Numerical, something like 1960, 1970, 1980, etc.
    - genre: Categorical, something like 'rock', 'pop', 'country', etc.
    - song_title: Text, the title of the song
    - artist_gender: Male or Female (or maybe we can add more categories later)
    - lyrics: Text, the lyrics of the song

We can add artist name, album name, and other metadata later if needed.

Data should be stored in CSV format in the data/ directory. We can use the pandas to do this.
'''
import pandas as pd

def extract_data_genius():
    '''
    Extract the data from Genius API. We can use the lyricsgenius package to do this.
    '''
    pass

def build_dataset():
    '''
    Use the extracted data to build the CSV dataset.
    '''
    pass

def extract_features(data_file):
    '''
    Extract features from the dataset to supply to the model.
    Basically converting the CSV to a numpy array or something the model can use.
    For the LSTM we'd probably have to use a tokenizer to convert the lyrics to a sequence of integers.
    I think PyTorch has a tokenizer we can use.
    '''
    pass

