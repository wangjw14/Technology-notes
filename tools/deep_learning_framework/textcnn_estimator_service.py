import tensorflow as tf
import pandas as pd
import os
from textcnn_estimator import params, DataProcessor

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

path_vocab = '../data//textcnn_estimator_vocab.txt'
path_dev = '../data/dev.tsv'

processor = DataProcessor(path_vocab,'',params)

predictor = tf.contrib.predictor.from_saved_model('export_model/1574155138')

df = pd.read_csv(path_dev,header=None,sep='\t')
df = df.sample(frac=1)[:10]
print(df)
df[1] = df[1].apply(lambda x:processor.process_text(x))

d = list(df[1].values)
# print(d)

r = predictor({"text_ids":d})
print(r)
print([processor.labels_rev_dict[i] for i in r['pred']])
