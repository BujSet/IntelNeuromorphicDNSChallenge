from os import listdir
from os.path import isfile, join
import sys, os
SRC_DIR = "/IntelNeuromorphicDNSChallenge/validation_set/clean/"
DST_DIR = "/IntelNeuromorphicDNSChallenge/validation_set/crepe_pitch_annotations/clean/"
srcFiles = [f for f in listdir(SRC_DIR) if isfile(join(SRC_DIR, f))]
dstFiles = [f for f in listdir(DST_DIR) if isfile(join(DST_DIR, f))]
dstFiles = set(dstFiles)
remaining = []

for srcFile in sorted(srcFiles):
    testName = srcFile.replace(".wav", ".f0.csv")
    if not testName in dstFiles:
        remaining.append( srcFile )
print("Found " + str(len(remaining)) + " remaining files to process")

for s in remaining:
    source = join(SRC_DIR, s)
    os.system('python3 -m crepe ' + source + ' --step-size 8 --model-capacity full --viterbi --output ' + DST_DIR)
