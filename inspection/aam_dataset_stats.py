from os import listdir
from os.path import isfile, join

mypath = "aam_audio_multitracks"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
allInstruments = set()
mappings = dict()
for filename in onlyfiles:
    tokens = filename.replace(".flac", "").split("_")
    mixID = int(tokens[0])
    instrument = tokens[1]
    allInstruments.add(instrument)
    if mixID in mappings.keys():
        mappings[mixID].add(instrument)
    else:
        mappings[mixID] = {instrument}

allInstruments = sorted(list(allInstruments))
# first we print the header
headerString = "MixID"
for instr in allInstruments: 
    headerString += "," + instr
print(headerString)
for mixID in sorted(mappings.keys()):
    csvString = str(mixID)
    for instr in allInstruments:
        if instr in mappings[mixID]:
            csvString += ",1.0"
        else:
            csvString += ",0.0"
    print(csvString)

