import os
import io
import sys
import numpy as np
import requests
import json
import operator
import argparse
import datetime
import traceback
import time
from tzlocal import get_localzone

import numpy as np
import rtspMonitor

import config as cfg
import model

RANDOM = np.random.RandomState(cfg.RANDOM_SEED)

BIRDWEATHER_ID = "99999"
BIRDWEATHER_LON = "00.000"
BIRDWEATHER_LAT = "00.000"

def clearErrorLog():

    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)

def writeErrorLog(msg):

    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(msg + '\n')

def loadCodes():

    with open(cfg.CODES_FILE, 'r') as cfile:
        codes = json.load(cfile)

    return codes

def loadLabels(labels_file):

    labels = []
    with open(labels_file, 'r') as lfile:
        for line in lfile.readlines():
            labels.append(line.replace('\n', ''))    

    return labels

def loadSpeciesList(fpath):

    slist = []
    if not fpath == None:
        with open(fpath, 'r') as sfile:
            for line in sfile.readlines():
                species = line.replace('\r', '').replace('\n', '')
                slist.append(species)

    return slist

def predictSpeciesList():

    l_filter = model.explore(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK)
    cfg.SPECIES_LIST_FILE = None
    cfg.SPECIES_LIST = []
    for s in l_filter:
        if s[0] >= cfg.LOCATION_FILTER_THRESHOLD:
            cfg.SPECIES_LIST.append(s[1])


def getSortedTimestamps(results):
    return sorted(results, key=lambda t: float(t.split('-')[0]))


def predict(samples):

    # Prepare sample and pass through model
    data = np.array(samples, dtype='float32')
    prediction = model.predict(data)

    # Logits or sigmoid activations?
    if cfg.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(np.array(prediction), sensitivity=-cfg.SIGMOID_SENSITIVITY)

    return prediction

def chunkToWav(sig):

    import soundfile as sf
    wav_io = io.BytesIO()
    sf.write(wav_io, sig, 48000, 'PCM_16',format='WAV')
    return wav_io.getbuffer().tobytes()

def monitorStream( cfg ):
    lstCommon = ["Cyanocitta cristata_Blue Jay", "Dryobates pubescens_Downy Woodpecker", "Dryobates villosus_Hairy Woodpecker","Poecile atricapillus_Black-capped Chickadee","Poecile hudsonicus_Boreal Chickadee","Sitta carolinensis_White-breasted Nuthatch"] 
    #lstCommon = ["Poecile atricapillus_Black-capped Chickadee"] 

    # Start time
    start_time = datetime.datetime.now()
    last_msg_time = time.time()
    last_bird = ""

    # Status
    print('Analyzing', flush=True)
    audio = rtspMonitor.AudioHandler()
    audio.start()
    #audio.mainloop()

    stream_end = time.time() + 24 * 60 * 60 * 7
    while time.time() < stream_end:

        # Open audio file and split into 3-second chunks
        #chunk_end = time.time() + 3
        #while time.time() < chunk_end:
        time.sleep(3.0)
        audio_frames = audio.getRawAudio()
        chunks = rtspMonitor.splitSignal(audio_frames, cfg.SAMPLE_RATE, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
        
        # Process each chunk
        try:
            start, end = 0, cfg.SIG_LENGTH
            results = {}
            samples = []
            timestamps = []
            
            for c in range(len(chunks)):
                # Add to batch
                samples.append(chunks[c])
                timestamps.append([start, end])

                # Check if batch is full or last chunk        
                if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                    continue

                # Predict
                p = predict(samples)

                soundscape_uploaded = False
                soundscape_id = ""

                dtm_obs = datetime.datetime.now() - datetime.timedelta(seconds=3)
                obs_date = dtm_obs.strftime("%Y-%m-%d")
                obs_time = dtm_obs.strftime("%H:%M:%S")
                obs_iso8601 = dtm_obs.astimezone(get_localzone()).isoformat()

                # Add to results
                for i in range(len(samples)):

                    # Get timestamp
                    s_start, s_end = timestamps[i]

                    # Get prediction
                    pred = p[i]

                    # Assign scores to labels
                    p_labels = dict(zip(cfg.LABELS, pred))

                    #HUMAN_DETECTED = False

                    # Catch if Human is recognized
                    #for x in range(len(p_labels)):
                    #    if "Human" in p_labels[x][0]:
                    #        HUMAN_DETECTED = True

                    # If human detected set all detections to human to make sure voices are not saved
                    #if HUMAN_DETECTED is True:
                    #    p_labels = [('Human_Human', 0.0)] * 10

                    # Sort by score
                    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

                    # Print top 5 results and advance indicies
                    for p_bird in p_sorted:
                        if p_bird[1] > cfg.MIN_CONFIDENCE and p_bird[0] in cfg.CODES and (p_bird[0] in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
                            label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(p_bird[0])]
                            resultMessage = '\n{}\t{}\t{:2.2f}%\n'.format(
                                dtm_obs.strftime("%H:%M:%S"),
                                label.split('_')[1], 
                                100*p_bird[1])
                            #if last_bird != p_bird[0] or time.time() > last_msg_time + 30:
                            print( resultMessage, flush=True)
                            last_msg_time = time.time()
                            last_bird = p_bird[0]
                            if p_bird[0] not in lstCommon:
                                rare_name = '{}_{}_{:.3f}.wav'.format(dtm_obs.strftime("%Y_%m_%d_%H_%M_%S"),p_bird[0],p_bird[1])
                                rare_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0]))), rare_name)
                                rtspMonitor.saveSignal(chunks[c], rare_path)
                            if BIRDWEATHER_ID != "99999":
                                #try:
                                if soundscape_uploaded is False:
                                    # POST soundscape to server
                                    soundscape_url = 'https://app.birdweather.com/api/v1/stations/' + \
                                        BIRDWEATHER_ID + \
                                        '/soundscapes' + \
                                        '?timestamp=' + \
                                        obs_iso8601

                                    # Convert chunk to wav format (16 bits)
                                    #wav_data = np.array(chunks[c], dtype=np.int16).tobytes()
                                    wav_data = chunkToWav( chunks[c] )

                                    response = requests.post(url=soundscape_url, data=wav_data, headers={'Content-Type': 'application/octet-stream'})
                                    print("Soundscape POST Response Status - ", response.status_code)
                                    sdata = response.json()
                                    soundscape_id = sdata['soundscape']['id']
                                    soundscape_uploaded = True

                                # POST detection to server
                                detection_url = "https://app.birdweather.com/api/v1/stations/" + BIRDWEATHER_ID + "/detections"
                                post_begin = "{ "
                                post_timestamp = "\"timestamp\": \"" + obs_iso8601 + "\","
                                post_lat = "\"lat\": " + BIRDWEATHER_LAT + ","
                                post_lon = "\"lon\": " + BIRDWEATHER_LON + ","
                                post_soundscape_id = "\"soundscapeId\": " + str(soundscape_id) + ","
                                post_soundscape_start_time = "\"soundscapeStartTime\": 0.0,"
                                post_soundscape_end_time = "\"soundscapeEndTime\": 3.0,"
                                post_commonName = "\"commonName\": \"" + label.split('_')[1] + "\","
                                post_scientificName = "\"scientificName\": \"" + label.split('_')[0] + "\","
                                post_algorithm = "\"algorithm\": " + "\"alpha\"" + ","
                                post_confidence = "\"confidence\": " + str(p_bird[1])
                                post_end = " }"

                                post_json = post_begin + post_timestamp + post_lat + post_lon + post_soundscape_id + post_soundscape_start_time + \
                                    post_soundscape_end_time + post_commonName + post_scientificName + post_algorithm + post_confidence + post_end
                                print(post_json)
                                response = requests.post(detection_url, json=json.loads(post_json))
                                print("Detection POST Response Status - ", response.status_code)
                                #except BaseException:
                                #    print("Cannot POST right now")

                # Clear batch
                samples = []
                timestamps = []  
        except:
            # Print traceback
            print(traceback.format_exc(), flush=True)

            # Write error log
            msg = 'Error: Cannot analyze audio .\n{}'.format(traceback.format_exc())
            print(msg, flush=True)
            writeErrorLog(msg)
            return False     

    audio.stop()

    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print('Finished in {:.2f} seconds'.format(delta_time), flush=True)

    return True

if __name__ == '__main__':

    # Clear error log
    #clearErrorLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default='example/', help='Path to input file or folder. If this is a file, --o needs to be a file too.')
    parser.add_argument('--o', default='example/', help='Path to output file or folder. If this is a file, --i needs to be a file too.')
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.')
    parser.add_argument('--slist', default='', help='Path to species list file or folder. If folder is provided, species list needs to be named \"species_list.txt\". If lat and lon are provided, this list will be ignored.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to 0.0.')
    parser.add_argument('--rtype', default='table', help='Specifies output format. Values in [\'table\', \'audacity\', \'r\', \'csv\']. Defaults to \'table\' (Raven selection table).')
    parser.add_argument('--batchsize', type=int, default=1, help='Number of samples to process at the same time. Defaults to 1.')
    parser.add_argument('--locale', default='en', help='Locale for translated species common names. Values in [\'af\', \'de\', \'it\', ...] Defaults to \'en\'.')
    parser.add_argument('--sf_thresh', type=float, default=0.03, help='Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to 0.03.')

    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    cfg.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.MODEL_PATH)
    cfg.LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.LABELS_FILE)
    cfg.TRANSLATED_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.TRANSLATED_LABELS_PATH)
    cfg.MDATA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.MDATA_MODEL_PATH)
    cfg.CODES_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.CODES_FILE)
    cfg.ERROR_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.ERROR_LOG_FILE)

    # Load eBird codes, labels
    cfg.CODES = loadCodes()
    cfg.LABELS = loadLabels(cfg.LABELS_FILE)

    # Load translated labels
    lfile = os.path.join(cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace('.txt', '_{}.txt'.format(args.locale)))
    if not args.locale in ['en'] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = loadLabels(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS   

    ### Make sure to comment out appropriately if you are not using args. ###

    # Load species list from location filter or provided list
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = args.lat, args.lon, args.week
    cfg.LOCATION_FILTER_THRESHOLD = max(0.01, min(0.99, float(args.sf_thresh)))
    if cfg.LATITUDE == -1 and cfg.LONGITUDE == -1:
        if len(args.slist) == 0:
            cfg.SPECIES_LIST_FILE = None
        else:
            cfg.SPECIES_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), args.slist)
            if os.path.isdir(cfg.SPECIES_LIST_FILE):
                cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, 'species_list.txt')
        cfg.SPECIES_LIST = loadSpeciesList(cfg.SPECIES_LIST_FILE)
    else:
        predictSpeciesList()
    if len(cfg.SPECIES_LIST) == 0:
        print('Species list contains {} species'.format(len(cfg.LABELS)))
    else:        
        print('Species list contains {} species'.format(len(cfg.SPECIES_LIST)))

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    # Set sensitivity
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(args.sensitivity) - 1.0), 1.5))

    # Set overlap
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))

    # Set result type
    cfg.RESULT_TYPE = args.rtype.lower()    
    if not cfg.RESULT_TYPE in ['table', 'audacity', 'r', 'csv']:
        cfg.RESULT_TYPE = 'table'

    # Set batch size
    cfg.BATCH_SIZE = max(1, int(args.batchsize))

    monitorStream( cfg )


    # A few examples to test
    # python3 analyze.py --i example/ --o example/ --slist example/ --min_conf 0.5 --threads 4
    # python3 analyze.py --i example/soundscape.wav --o example/soundscape.BirdNET.selection.table.txt --slist example/species_list.txt --threads 8
    # python3 analyze.py --i example/ --o example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0 --rtype table --locale de
    
