'''
In the paper they use a CNN over spectrograms, but we can use
an LSTM over the lyrics and classify [artist/genre/decade]. We'll use
the weight matrix from the last layer of the classification as the embedding.
'''