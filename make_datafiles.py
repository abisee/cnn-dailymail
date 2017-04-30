import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"

VOCAB_SIZE = 200000

def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print "Preparing to tokenize %s..." % stories_dir
  stories = os.listdir(stories_dir)
  # make IO list file
  print "Making list of files to tokenize..."
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print "Tokenizing %s..." % (stories_dir)
  subprocess.call(command)
  print "Finished tokenizing!"
  os.remove("mapping.txt")


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def write_to_bin(url_file, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print "Making bin file for %s..." % url_file
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))

      # Find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
        story_file = os.path.join(cnn_stories_dir, s)
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_stories_dir, s)
      else:
        raise Exception("can't find story file")

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print "Finished writing file %s" % out_file

  # write vocab to file
  if makevocab:
    print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file"



if __name__ == '__main__':
  if len(sys.argv) != 3:
    print "USAGE: python stories_to_bin.py <cnn_stories_dir> <dailymail_stories_dir>"
    sys.exit()
  cnn_stories_dir = sys.argv[1]
  dm_stories_dir = sys.argv[2]

  if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
  if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  print "cnn stories dir: ", cnn_stories_dir
  print "dm stories dir: ", dm_stories_dir

  # Run stanford tokenizer on both stories dirs
  tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
  tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
