import nltk
# You might need to download the Penn Treebank samples for the first run if you haven't already
# nltk.download('treebank')
from nltk.corpus import treebank

from ptb_store_corpus import train_startidx, dev_startidx, test_startidx

def write_embeddings_to_file(out_file, split='train'):
    fileids = []
    if split == 'train':
        fileids = treebank.fileids()[:dev_startidx]
    elif split == 'dev':
        fileids = treebank.fileids()[dev_startidx:test_startidx]
    else:
        assert split == 'test'
        fileids = treebank.fileids()[test_startidx:]

    

    

    

if __name__=="__main__":
    write_embeddings_to_file('example/data/ptb_nltk/ptb_train.elmo-layers.hdf5', 'train')
    # write_embeddings_to_file('example/data/ptb_nltk/ptb_dev.elmo-layers.hdf5', 'dev')
    # write_embeddings_to_file('example/data/ptb_nltk/ptb_test.elmo-layers.hdf5', 'test')