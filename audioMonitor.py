import numpy as np
import pyaudio
import time
import librosa

import config as cfg

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 1024
        self.p = None
        self.stream = None
        self.audio_frames = []
        self.input_device = 2
        

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK,
                                  input_device_index = self.input_device)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def getRawAudio(self):
        frames = self.audio_frames
        self.audio_frames = []
        numpy_array = np.concatenate( frames )
        return np.concatenate( frames )

    def callback(self, in_data, frame_count, time_info, flag):
        self.audio_frames.append( np.frombuffer(in_data, dtype=np.float32) )
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(2.0)

def saveSignal(sig, fname):

    import soundfile as sf
    sf.write(fname, sig, 48000, 'PCM_16')

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype('float32')

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))
        
        sig_splits.append(split)

    return sig_splits