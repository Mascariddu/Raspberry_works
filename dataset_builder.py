import pandas as pd
import numpy as np
import tensorflow as tf
import os
import io
import sys

from datetime import datetime
import argparse

# python3 dataset_builder.py -i ./raw_data -o ./fusion.tfrecord

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input path', required=True)
parser.add_argument('-o', type=str, help='output path', required=True)
args = parser.parse_args()

input_path = args.i
output_path = args.o

samples = pd.read_csv(input_path+'/samples.csv',header=None)

with tf.io.TFRecordWriter(output_path) as writer:
    for i in range(10):
        timestamp = int(datetime.strptime(samples.iloc[i,0]+samples.iloc[i,1],'%d/%m/%Y%H:%M:%S').timestamp())
        datetime_ = tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp]))

        t = samples.iloc[i,2].astype(np.int32)
        h = samples.iloc[i,3].astype(np.int32)
        
        temp = tf.train.Feature(int64_list=tf.train.Int64List(value=[t]))
        hum = tf.train.Feature(int64_list=tf.train.Int64List(value=[h]))

        raw_audio = tf.io.read_file(input_path+f'/{samples.iloc[i,4]}')
        audio = raw_audio.numpy()
        audio = tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio]))
                
        mapping = {'datetime': datetime_,
                   'temperature': temp,
                   'humidity': hum,
                   'audio': audio
                   }
        
        record = tf.train.Example(features=tf.train.Features(feature=mapping))
        serialized = record.SerializeToString()
        
        writer.write(serialized)

writer.close()
