from os import listdir
from os.path import isfile, join
import numpy as np 
import sys

def getScores(results_dir):
    files = [f for f in listdir(results_dir) if isfile(join(results_dir, f))]
    scores = np.empty([50, 50], dtype=float)
    minScore = 1000.0
    maxScore = 0.0
    dists = np.zeros((len(files),), dtype=float)
    flattened = np.zeros((len(files),), dtype=float)
    for i in range(len(files)):
        filename = files[i]
        path = join(results_dir, filename)
        name_tokens = filename.replace('.txt', '').split('_')

        # Make sure but speech and noise are from same channel
        assert(int(name_tokens[-1]) == int(name_tokens[-3]))

        try:
            speech = int(name_tokens[-4])
            noise = int(name_tokens[-2])
        except:
            print("Error, standard format not met\n")
            print(filename)
            print(name_tokens)
            sys.exit(1)
        with open(path, 'r') as F:
            lines = F.readlines()
            tokens = lines[-1].split()
            score = -1.0
            for line in lines:
                if "Valid" in line and "SI-SNR" in line and "dB" in line:
                    tokens = line.split()
                    score = float(tokens[-2])
                    dists[i] = ((speech - noise) * 5.625)
                    if (score > maxScore):
                        maxScore = score
                    if (score < minScore):
                        minScore = score
#                    print("Found " + str(score) + " for (" + str(speech) + "," + str(noise) + ")")
                    break

            scores[speech, noise] = score
            flattened[i] = score
    return (maxScore, minScore, scores, dists, flattened)
