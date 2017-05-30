This code produces the non-anonymized version of the CNN / Daily Mail summarization dataset, as used in the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)*. It processes the dataset into the binary format expected by the [code](https://github.com/abisee/pointer-generator) for the Tensorflow model.

# Instructions

## 1. Download data
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. 

**Warning:** These files contain a few (114, in a dataset of over 300,000) examples for which the article text is missing - see for example `cnn/stories/72aba2f58178f2d19d3fae89d5f3e9a4686bc4bb.story`. The [Tensorflow code](https://github.com/abisee/pointer-generator) has been updated to discard these examples.

## 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2016-10-31` directory. You can check if it's working by running
```
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer
```
You should see something like:
```
Please
tokenize
this
text
.
PTBTokenizer tokenized 5 tokens at 68.97 tokens per second.
```
## 3. Process into .bin and vocab files
Run
```
python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories
```
replacing `/path/to/cnn/stories` with the path to where you saved the `cnn/stories` directory that you downloaded; similarly for `dailymail/stories`.

This script will do several things:
* The directories `cnn_stories_tokenized` and `dm_stories_tokenized` will be created and filled with tokenized versions of `cnn/stories` and `dailymail/stories`. This may take some time. ***Note**: you may see several `Untokenizable:` warnings from Stanford Tokenizer. These seem to be related to Unicode characters in the data; so far it seems OK to ignore them.*
* For each of the url lists `all_train.txt`, `all_val.txt` and `all_test.txt`, the corresponding tokenized stories are read from file, lowercased and written to serialized binary files `train.bin`, `val.bin` and `test.bin`. These will be placed in the newly-created `finished_files` directory. This may take some time.
* Additionally, a `vocab` file is created from the training data. This is also placed in `finished_files`.
* Lastly, `train.bin`, `val.bin` and `test.bin` will be split into chunks of 1000 examples per chunk. These chunked files will be saved in `finished_files/chunked` as e.g. `train_000.bin`, `train_001.bin`, ..., `train_287.bin`. This should take a few seconds. You can use either the single files or the chunked files as input to the Tensorflow code (see considerations [here](https://github.com/abisee/cnn-dailymail/issues/3)).
