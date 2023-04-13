import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from lyricsgenius import Genius
genius = Genius("XN6f6ECwHaLNf0dun2Sobkx6bp5ZwoIllko41uM-2qb7ONKqGpx4rKunAsVwtvcS", timeout=100, skip_non_songs = True)

class Dataset:
    def __init__(self, data_file, max_len=512):
        self.max_len = max_len
        self.labels = ['pop female', 'pop male', 
                       'rock female', 'rock male', 
                       'rap female', 'rap male',
                       'country female', 'country male']
        self.labels_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_labels = {i: label for i, label in enumerate(self.labels)}

        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)

        self.x, self.y = self.process(pd.read_csv(data_file))
    
    def __len__(self):
        return len(self.data)
    
    def process(self, df):
        x = np.zeros((len(df), self.max_len), dtype=np.int64)
        y = np.zeros((len(df)), dtype=np.int64)
        for i in range(len(df)):
            row = df.iloc[i]
            x[i, :] = self.encode_text(row['lyrics'])
            y[i] = self.labels_to_idx[row['genre'] + ' ' + row['gender']]
        return x, y
    
    def encode_text(self, text):
        return self.tokenizer.encode(text).ids

    def decode_text(self, ids):
        return self.tokenizer.decode(ids)
    
def load_dataset(save_path):
    data = {'genre': [], 'gender': [], 'lyrics': []}
    pop_female_artists = ['Adele', 'Ariana Grande', 'Beyoncé', 'Billie Eilish', 'Britney Spears', 'Christina Aguilera', 'Dua Lipa', 'Halsey', 'Jennifer Lopez', 'Katy Perry', 'Lady Gaga', 'Demi Lovato', 'Pussycat Dolls', 'Miley Cyrus', 'P!nk', 'Rihanna', 'Selena Gomez', 'Shakira', 'Kelly Clarkson', 'Gwen Stefani']
    for female_artist in pop_female_artists:
        try:
            artist = genius.search_artist(female_artist, max_songs=1)
        except:
            print('Error searching {}'.format(female_artist))
            continue
        songs = artist.songs
        for song in songs:
            data['lyrics'].append(song.lyrics)
            data['genre'].append('pop')
            data['gender'].append('female')
            print('Added {} to dataset'.format(song.full_title))

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    return df

# load_dataset('./data/data.csv')
dataset = Dataset('./data/data.csv')
print(dataset.x.shape)
print(dataset.x[0])
print(dataset.y[0])
print(dataset.decode_text(dataset.x[0]))