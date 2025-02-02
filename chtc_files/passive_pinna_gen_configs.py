from os import listdir, getcwd
from os.path import isfile, join
import re, sys

subjects = ["003"]
others = ["008","009","010","011",
"012","015","017","018","019",
"020","021","027","028","033",
"040","044","048","050","051",
"058","059","060","061","065",
"119","124","126","127","131",
"133","134","135","137","147",
"148","152","153","154","155",
"156","158","162","163","165"]

channels = [0] #[0, 1]


allConfigs = set()
speechDelta = 25
noiseDelta = 25
speechOrients = range(0, 1250, speechDelta)
for s in subjects:
    for c in channels:
        for so in speechOrients:
            for no in range(0, 1250, noiseDelta):
                allConfigs.add( (int(s),c,so,no) )
                configString = s + "," + str(c) + "," + str(so) + "," + str(no) + ","
                if (no + noiseDelta >= 1250):
                    configString += str(1250 - no)
                else:
                    configString += str(noiseDelta)
#                print(configString)
#sys.exit(0)

print(len(allConfigs))
doneDir = join(getcwd(),"done")
onlyfiles = [f for f in listdir(doneDir) if isfile(join(doneDir,f)) and ".out" in f]
#print(onlyfiles)
dataMatch = r"\d+,\d+,\d+,\d+,\d+\.\d+"
completedConfigs = dict()
for doneFile in onlyfiles:
    with open(join(doneDir, doneFile)) as f:
        lines = f.readlines()
        for line in lines:
            if (re.search(dataMatch, line)):
                tokens = line.split(",")
                sub = int(tokens[0])
                chan = int(tokens[1])
                speechO = int(tokens[2])
                noiseO = int(tokens[3])
                score = float(tokens[4])
                key = (sub, chan, speechO, noiseO)
                completedConfigs[key] = score # overrides duplicates?
onlyfiles = [f for f in listdir(getcwd()) if isfile(join(getcwd(),f)) and ".out" in f]
for newFile in onlyfiles:
    with open(join(getcwd(), newFile)) as f:
        lines = f.readlines()
        for line in lines:
            if (re.search(dataMatch, line)):
                tokens = line.split(",")
                sub = int(tokens[0])
                chan = int(tokens[1])
                speechO = int(tokens[2])
                noiseO = int(tokens[3])
                score = float(tokens[4])
                key = (sub, chan, speechO, noiseO)
                completedConfigs[key] = score # overrides duplicates?
print(len(completedConfigs.keys()))
print(len(allConfigs.difference(completedConfigs.keys())))
print(str(float(len(completedConfigs.keys()))/float(len(allConfigs))) + "% completed")

# collate results to output results.txt
'''
with open("results.txt", "w") as of:
    for key in sorted(completedConfigs.keys()):
        sub,chan,so,no = key
        val = completedConfigs[key]
        line = str(sub) + ","
        line += str(chan) + ","
        line += str(so) + ","
        line += str(no) + ","
        line += str(val) + "\n"
        of.write(line)
'''

                
