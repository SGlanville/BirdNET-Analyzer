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
        self.CHUNK = 4096
        self.t = None
        self.stream = None
        self.audio_frames = []
        self.is_active = True

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
            if not self.is_active: 
                break

	# Launches the video recording function using a thread			
    def start(self):
        self.t = threading.Thread(target=self.listen)
        self.t.start()

    def stop(self):
        self.is_active = False

    def getRawAudio(self):
        if not self.t.is_alive():
            self.start()
        frames = self.audio_frames
        self.audio_frames = []
        if len(frames) <= 1:
            return frames
        else:
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

def latestSignal(sig, seconds, minSeconds):
    chunkLen = int(48000 * seconds)
    siglen = len(sig)
    # Split signal with overlap
    if siglen < int(minSeconds * 48000):
        return []
    if siglen > chunkLen:
        return sig
    if siglen > chunkLen:
        return sig[-chunkLen:]

    return np.hstack((sig, noise(sig, chunkLen - siglen, 0.5)))
