import os
from os import listdir
from os.path import isfile, join, getsize
mypath = os.getcwd()
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
scores = []
for filename in onlyfiles:
    if ".out" in filename:
        tokens = filename.split(".")
        tokens = tokens[0].split("_")
        numEpochs = int(tokens[1])
        sisnrScore = None
        with open(join(mypath, filename)) as f:
            lines = [line.rstrip() for line in f]
            for line in lines:
                if ("Final validation score: " in line):
                    values = line.split()
                    score = float(values[3])
                    sisnrScore = score
                    break
        scores.append( (numEpochs, sisnrScore) )
for e,s in sorted(scores):
    print(str(e) + "\t" + str(s))
