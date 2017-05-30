# Rumor Dataset

Dataset for detecting rumor on [Twitter](https://twitter.com)

## Prerequisites
* [Tensorflow](https://www.tensorflow.org)
* [TextBlob](https://textblob.readthedocs.io)

## Usage

Build TFRecord file for [Tensorflow](https://www.tensorflow.org)

```bash
$ python build_dataset.py \
--dataset_directory=/your/dataset/directory \
--output_directory=/your/output/directory \
--num_threads=8
```