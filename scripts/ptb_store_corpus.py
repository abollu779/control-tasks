import nltk
# You might need to download the Penn Treebank samples for the first run if you haven't already
# nltk.download('treebank')
from nltk.corpus import treebank

train_startidx, dev_startidx, test_startidx = 0, 150, 180

def write_dataset_to_file(out_file, split='train'):
    fileids = []
    if split == 'train':
        fileids = treebank.fileids()[:dev_startidx]
    elif split == 'dev':
        fileids = treebank.fileids()[dev_startidx:test_startidx]
    else:
        assert split == 'test'
        fileids = treebank.fileids()[test_startidx:]
    
    tagged_sents = []
    for filename in fileids:
        file_data = treebank.tagged_sents(filename) # outputs data from that file in the form [(word1, label1), (word2, label2),....]
        tagged_sents.extend(file_data)
        
    with open(out_file, 'w') as out_fp:
        for tagged_sent in tagged_sents:
            for i, tagged_word in enumerate(tagged_sent):
                out_fp.write('%d\t%s\t%s\n' % (i, tagged_word[0], tagged_word[1]))
            out_fp.write('\n')

if __name__=="__main__":
    write_dataset_to_file('example/data/ptb_nltk/ptb_train.txt', 'train')
    write_dataset_to_file('example/data/ptb_nltk/ptb_dev.txt', 'dev')
    write_dataset_to_file('example/data/ptb_nltk/ptb_test.txt', 'test')