
#TOGO:
# python3 HW1_ex5_Group9.py --numsamples 5 --output ./out
from subprocess import Popen
from subprocess import PIPE
import subprocess
import argparse
import os
import pyaudio
import time
import wave
import io
from scipy import signal
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

performance = ['sudo', 'sh', '-c','echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
powersave = ['sudo', 'sh', '-c','echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
x = Popen(powersave)
x.wait()

parser = argparse.ArgumentParser()
parser.add_argument('--numsamples', type=int, help='number of samples recorded', required=True)
parser.add_argument('--output', type=str, help='output path', required=True)
args = parser.parse_args()

# Parameters for recording
samp_rate = 48000
resolution = pyaudio.paInt16
chunk = 4800
record_secs = 1
dev_index = 0

# Parameter given by console
output_folder = args.output
n_samples = args.numsamples

#resampling frequency
new_rate = 16000

# stft params
frame_length = 640 # 0.04 * dim_audio
frame_step = 320 # 0.02 * dim_audio

min_ = -32768
max_ = -min_

# mfcc params
num_spectrogram_bins = 321
num_mel_bins = 40
lower_frequency = 20
upper_frequency = 4000
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                                        num_spectrogram_bins, new_rate,
                                                                        lower_frequency, upper_frequency)
p = Popen(['sudo', 'sh', '-c', 'echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset'], shell=False)
p.wait()
audio = pyaudio.PyAudio()
stream = audio.open(format=resolution, rate=samp_rate, channels=1,
                        input_device_index=dev_index, input=True, frames_per_buffer=chunk)

# Repeat the recording n_samples times:

for i in range(n_samples):
    start = time.time()

    stream.start_stream()

    buffer = io.BytesIO()
    for k in range(int((samp_rate / chunk) * record_secs)):
        if k==0:
            Popen(powersave)
        if k==9:
            Popen(performance)
        buffer.write(stream.read(chunk))
    
    stream.stop_stream()

    # Resampling 48kHz -> 16kHz
    sampled_audio = np.frombuffer(buffer.getvalue(), np.int16)   
    sampled_audio = signal.resample_poly(sampled_audio, 1, samp_rate/new_rate) # resampling signal
    tf_audio = tf.convert_to_tensor(sampled_audio, dtype=tf.float32)
    tf_audio = (2 * (tf_audio - min_)/(max_ - min_)) -1

    stft = tf.signal.stft(tf_audio, frame_length=640, frame_step=320,
                          fft_length=640)
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix,1)

    mel_spectrogram.set_shape((49,40))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:,:10]

    tensor = tf.io.serialize_tensor(mfccs)
    tf.io.write_file('/home/pi/WORK_DIR/HW1/'+output_folder+'/mfccs'+str(i)+'.bin',tensor)

    print(time.time()-start)

p = Popen(powersave)
stream.close()
audio.terminate()

Popen(['cat', '/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'])
