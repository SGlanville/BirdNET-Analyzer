import numpy as np
import time
import ffmpeg
import threading

import config as cfg

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)

class AudioHandler(object):
    def __init__(self):
        self.INPUT = 'rtsps://192.168.1.1:7441/DkIFZ0lKWoboo5wl?enableSrtp'
        self.FORMAT = 'f32le'
        self.CODEC = 'pcm_f32le'
        self.CHANNELS = 1
        self.RATE = '48k'
        self.CHUNK = 1024
        self.t = None
        self.stream = None
        self.audio_frames = []

    def listen(self):
        self.stream = (
            ffmpeg
            .input(self.INPUT)
            .output('-', format=self.FORMAT, acodec=self.CODEC, ac=self.CHANNELS, ar=self.RATE)
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )
        while self.stream.poll() is None:
            in_data = self.stream.stdout.read(self.CHUNK)
            self.audio_frames.append( np.frombuffer(in_data, dtype=np.float32) )

	# Launches the video recording function using a thread			
    def start(self):
        self.t = threading.Thread(target=self.listen)
        self.t.start()

    def stop(self):
        self.t.stop() 

    def getRawAudio(self):
        frames = self.audio_frames
        self.audio_frames = []
        numpy_array = np.concatenate( frames )
        return np.concatenate( frames )

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