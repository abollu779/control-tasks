from collections import namedtuple
import nltk
import os
from torch.utils.data import DataLoader, Dataset

# You might need to download the Penn Treebank samples for the first run if you haven't already
# nltk.download('treebank')

from nltk.corpus import treebank

# Create a PTBObservationIterator(Dataset) and implement __init__, __len__, __getitem__

# Create a PTBDataset class and implement __init__, get_train_dataloader, get_dev_dataloader, 
# get_test_dataloader, custom_pad (collate_fn for Dataloader)

class SimpleDataset:
  """
  Stores PTB sentences and POS labels in namedtuples called Observations
  """
  def __init__(self, args, task):
    self.args = args
    self.batch_size = args['dataset']['batch_size']
    self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    train_obs, dev_obs, test_obs = self.read_from_disk()

  def read_from_disk(self):
    '''Reads observations from stored txt files
    as specified by the yaml arguments dictionary and 
    optionally adds pre-constructed embeddings for them.

    Returns:
    A 3-tuple: (train, dev, test) where each element in the
    tuple is a list of Observations for that split of the dataset. 
    '''
    train_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['train_path'])
    dev_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['dev_path'])
    test_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['test_path'])
    train_observations = self.load_ptb_dataset(train_corpus_path)
    dev_observations = self.load_ptb_dataset(dev_corpus_path)
    test_observations = self.load_ptb_dataset(test_corpus_path)
    import pdb
    pdb.set_trace()

  def generate_lines_for_sent(self, lines):
    buf = []
    for line in lines:
      if not line.strip(): # Encounted newline
        if buf:
          yield buf
          buf = []
        else:
          continue
      else:
        buf.append(line.strip())
    if buf:
      yield buf


  def load_ptb_dataset(self, filepath):
    '''Reads in a txt ptb data file; generates Observation objects

    For each sentence, generates a single Observation object.

    Args:
      filepath: the filesystem path to the conll dataset
  
    Returns:
      A list of Observations 
    '''
    observations = []
    lines = (x for x in open(filepath))
    for buf in self.generate_lines_for_sent(lines):
      ptb_lines = []
      for line in buf:
        ptb_lines.append(line.strip().split('\t'))
      embeddings = [None for x in range(len(ptb_lines))]
      observation = self.observation_class(*zip(*ptb_lines), embeddings)
      observations.append(observation)
    return observations

  def get_observation_class(self, fieldnames):
    '''Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
        fieldnames: a list of strings corresponding to the information about each
        sentence to be captured in this namedtuple.
    Returns:
        A namedtuple class; each observation in the dataset will be an instance
        of this class.
    '''
    return namedtuple('Observation', fieldnames)

class ELMoDataset(SimpleDataset):
  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    return observations
