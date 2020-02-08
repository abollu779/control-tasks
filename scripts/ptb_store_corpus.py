import nltk
# You might need to download the Penn Treebank samples for the first run if you haven't already
# nltk.download('treebank')
from nltk.corpus import treebank

"""
Run this script to store the Penn Treebank sentences in a raw .txt file and 
the formatted tagged sentences in a .conllu file.

In order to generate ELMo embeddings, make sure you have allennlp installed and 
run the following from the command line when in the root directory:
$ allennlp elmo --all example/data/ptb_nltk/ptb_train.txt example/data/ptb_nltk/ptb_train.elmo-layers.hdf5 --cuda-device 0
$ allennlp elmo --all example/data/ptb_nltk/ptb_dev.txt example/data/ptb_nltk/ptb_dev.elmo-layers.hdf5 --cuda-device 0
$ allennlp elmo --all example/data/ptb_nltk/ptb_test.txt example/data/ptb_nltk/ptb_test.elmo-layers.hdf5 --cuda-device 0
"""
train_startidx, dev_startidx, test_startidx = 0, 150, 180

def write_dataset_to_file(out_conll_file, out_txt_file, split='train'):
    fileids = []
    if split == 'train':
        fileids = treebank.fileids()[:dev_startidx]
    elif split == 'dev':
        fileids = treebank.fileids()[dev_startidx:test_startidx]
    else:
        assert split == 'test'
        fileids = treebank.fileids()[test_startidx:]
    
    # Collect corpus data
    tagged_sents = []
    sents = []
    for filename in fileids:
        file_tagged_data = treebank.tagged_sents(filename) 
        # outputs data from that file in the form:
        # [[(word1_sent1, label1_sent1), (word2_sent1, label2_sent1),..],[(word1_sent2,label1_sent2),...],...]
        tagged_sents.extend(file_tagged_data)

        file_sents = treebank.sents(filename)
        # outputs data from that file in the form:
        # [[word1_sent1, word2_sent1,..],[word1_sent2,...],...]
        sents.extend(file_sents)

    # Write conll file
    with open(out_conll_file, 'w') as out_conll_fp:
        for tagged_sent in tagged_sents:
            for i, tagged_word in enumerate(tagged_sent):
                out_conll_fp.write('%d\t%s\t%s\n' % (i, tagged_word[0], tagged_word[1]))
            out_conll_fp.write('\n')

    # Write raw txt file
    with open(out_txt_file, 'w') as out_txt_fp:
        for sent in sents:
            out_txt_fp.write(' '.join(sent))
            out_txt_fp.write('\n')


if __name__=="__main__":
    write_dataset_to_file('example/data/ptb_nltk/ptb_train.conllu', 'example/data/ptb_nltk/ptb_train.txt', 'train')
    write_dataset_to_file('example/data/ptb_nltk/ptb_dev.conllu', 'example/data/ptb_nltk/ptb_dev.txt','dev')
    write_dataset_to_file('example/data/ptb_nltk/ptb_test.conllu', 'example/data/ptb_nltk/ptb_test.txt','test')
