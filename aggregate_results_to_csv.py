#!/usr/bin/env python3
import os
import csv
import re
import sys

# Directories to search (non-recursive)
TARGET_DIRS = [
    "/home/selagamsetty/ndns/passive_pinna_on_validation_set",
    "/home/selagamsetty/ndns/passive_pinna_on_validation_set/chechil_passive_pinna",
    "/home/selagamsetty/ndns/passive_pinna_on_validation_set/cpu_trials",
    "/home/selagamsetty/ndns/passive_pinna_on_validation_set/done"
]

# Output file
OUTPUT_CSV = "collated_results.csv"

COLUMN_HEADERS = [
    "Subject",
    "Channel",
    "Speech Orient",
    "Noise Orient",
    "Final Validation Score SI-SNR (dB)"
        ]

class OutputResult:

    def __init__(self, sub, chan, spOrient, noOrient, sisnr):
        try:
            self.subject = int(sub)
            self.channel = int(chan)
            self.speechOrient = int(spOrient)
            self.noiseOrient = int(noOrient)
            self.sisnr = float(sisnr)
        except:
            print(f"Error: Input string cannot be converted to an data types.")

    def __hash__(self):
        return hash((self.subject, self.channel, self.speechOrient, self.noiseOrient))

    def __eq__(self, other):
        if not isinstance(other, OutputResult):
            return False
        if self.subject != other.subject:
            return False
        if self.channel != other.channel:
            return False
        if self.speechOrient != other.speechOrient:
            return False
        if self.noiseOrient != other.noiseOrient:
            return False
        #if abs(self.sisnr - other.sisnr) > 0.0001:
        #    return False
        return True

    def toString(self):
        result = ""
        result += str(self.subject) + ","
        result += str(self.channel) + ","
        result += str(self.speechOrient) + ","
        result += str(self.noiseOrient) + ","
        result += str(self.sisnr) + "\n"
        return result

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()

    def __lt__(self, other):
        if self.subject != other.subject:
            return self.subject < other.subject
        if (self.channel != other.channel):
            return self.channel < other.channel
        if (self.speechOrient != other.speechOrient):
            return self.speechOrient < other.speechOrient
        if (self.noiseOrient != other.noiseOrient):
            return self.noiseOrient < other.noiseOrient
        return True


def line_contains_needed_headers(line):
    for cHeader in COLUMN_HEADERS:
        if not (cHeader in line):
            return False
    return True

def parse_file(filename, lines):
    headerIdx = -1
    for i  in range(len(lines)):
        if (line_contains_needed_headers(lines[i])):
            headerIdx = i
            break
    if headerIdx < 0:
        return None
#    print(f"headerIdx = {headerIdx}")
    headerLineHeaders = lines[headerIdx].strip().split(",")
    column_indicies = [-1 for _ in COLUMN_HEADERS]
    for i in range(len(COLUMN_HEADERS)):
        cHeader = COLUMN_HEADERS[i]
        for j in range(len(headerLineHeaders)):
            headerLineHeader = headerLineHeaders[j]
            if cHeader.strip() == headerLineHeader.strip():
                column_indicies[i] = j
                break
#    print(column_indicies)
    for cIdx in column_indicies:
        assert(cIdx >= 0);
        assert(cIdx < len(headerLineHeaders))

        
    dataLines = lines[headerIdx+1:]
    results = set()
    for dLine in dataLines:
        dataValues = dLine.split(",")
        colVals = []
        for cIdx in column_indicies:
            colVals.append(dataValues[cIdx])

        result = OutputResult(colVals[0],
                     colVals[1],
                     colVals[2],
                     colVals[3],
                     colVals[4])
#        if colVals[2] == "0" and colVals[3] == "0":
#            print(filename)
#            print(result)
        if not result in results:
            results.add(result)
    return results

def collect_results():
    """Scan each target directory (non-recursive) and parse .log/.out files."""
    allResults = set()

    numLogFilesParsed = 0
    numLogFilesWithData = 0
    numOutFilesParsed = 0
    numOutFilesWithData = 0
    numTotalFilesParsed = 0
    numComputedResults = 0
    for directory in TARGET_DIRS:
#        print(f"Looking at files from directory: {directory}")
        try:
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".log") or filename.endswith(".out"):
                    filepath = os.path.join(directory, filename)
                    with open(filepath, "r", errors="ignore") as f:
                        parsed = parse_file(filename, f.readlines())
                    if filename.endswith(".log"):
                        numLogFilesParsed += 1
                        if parsed != None:
                            numLogFilesWithData += 1
                    else:
                        assert(filename.endswith(".out"))
                        numOutFilesParsed += 1
                        if parsed != None:
                            numOutFilesWithData += 1

                    if parsed: 
                        numComputedResults += len(parsed)
                        for parsedResult in parsed:
                            if not parsedResult in allResults:
                                allResults.add(parsedResult)
#                        allResults |= parsed
                numTotalFilesParsed = numLogFilesParsed + numOutFilesParsed
#                if (numTotalFilesParsed > 8):
#                    break
        except FileNotFoundError:
            print(f"Warning: Directory not found: {directory}")
    print("Statistics:")
    print(f"Total Number of Files Parsed: {numTotalFilesParsed}")
    print(f"\tLog files (total): {numLogFilesParsed}")
    print(f"\tLog files (with data): {numLogFilesWithData}")
    print(f"\tOut files (total): {numOutFilesParsed}")
    print(f"\tOut files (with data): {numOutFilesWithData}")
    print(f"Total number of data points computed by all jobs: {numComputedResults}")
    print(f"Total number of unique data points: {len(allResults)}")
    duplicateJobPercentage = 100.0 * ((1.0 * (numComputedResults - len(allResults))) / len(allResults))
    print(f"Fraction of data points that computed redundantly: {duplicateJobPercentage}%")
    return allResults

def write_csv(results):
    """Write all parsed results to CSV, overwriting if it exists."""
    with open(f"{OUTPUT_CSV}", "w") as of:
        for i in range(len(COLUMN_HEADERS)):
            if i == len(COLUMN_HEADERS) - 1:
                of.write(COLUMN_HEADERS[i] + "\n")
            else:
                of.write(COLUMN_HEADERS[i] + ",")

        for oR in sorted(results):
            of.write(str(oR))

def main():
    results = collect_results()
    if not results:
        print("No matching results found in the specified directories.")
    else:
        write_csv(results)
        print(f"✅ Collated {len(results)} entries into {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

