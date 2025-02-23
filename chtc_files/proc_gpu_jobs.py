import os, sys
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta

directories = [
        "ntp640", 
        "ndns/epoch05", 
        "ndns/epoch10",
        "ndns/epoch25",
        "ndns/epoch50",
        "ndns/latest05", "ndns/latest10", "ndns/latest25", "ndns/latest50",
        "ndns/h100_test",
        "ndns/sweep_el/far_right/",
        "ndns/sweep_el/midsagittal/",
        "ndns/sweep_epochs/cosine_annealing_LR/fixed_orients_1/done",
        "ndns/sweep_epochs/cosine_annealing_LR/fixed_orients_2/done",
        "ndns/sweep_epochs/cosine_annealing_LR/with_cipic/done",
        "ndns/sweep_epochs/cosine_annealing_LR/without_cipic",
        "ndns/sweep_epochs/cosine_annealing_LR/without_cipic/done",
        "ndns/sweep_epochs/default/with_cipic/done",
        "ndns/sweep_epochs/default/without_cipic/done",
        "ndns/sweep_epochs/multi_step_LR_20_80/with_cipic/done",
        "ndns/sweep_epochs/multi_step_LR_20_80/without_cipic/done",
        "ndns/sweep_orients",
        "ndns/passive_pinna_on_validation_set",
        "ndns/pitch_predictor/snn/baseline",
        "ndns/pitch_predictor/snn/baseline/sweeps/depth",
        "ndns/pitch_predictor/snn/baseline/sweeps/depth/latest",
        "ndns/pitch_predictor/snn/baseline/sweeps/width",
        "ndns/pitch_predictor/snn/cipic/fixed_orient_1"
        ]

class JobStats:
    def __init__(self, logfile):
        pathTokens = logfile.split("/")
        pathTokens = pathTokens[:-1]
        self.logdir = "/".join(pathTokens) + "/"
        self.logfile = logfile
        self.submitTime = None
        self.inputFileTransferStartTime = None
        self.inputFileTransferEndTime = None
        self.outputFileTransferStartTime = None
        self.outputFileTransferEndTime = None
        self.jobExecutionStartTime = None
        self.jobTerminationTime = None
        self.jobID = None
        self.clusterID = None
        self.processID = None 
        self.slotName = None 
        self.machineName = None
        self.trainingEpochs = None
        self.validationEpochs = None
        self.slotName = None
        self.machineName = None
        self.deviceName = None
        self.timeExecuted = None
        self.timeSlotBusy = None
        self.ndnsDatasetTransferStart = None
        self.ndnsDatasetTransferEnd = None
        self.exitCode = None
        # in some cases, jobs finish fine but the stats are all mangled due to some error with the scheduler
        self.statsCoherent = True
        self.jobWasEvicted = False
        self.diskSpaceExceeded = False
        self.jobFailedToComplete = False
        self.jobWasAborted = False
        self.gpuJobLengthDesc = None
        self.gpuJobLength = None
        self.selfSchedulingEnabled = False
        self.gpuJobLengthEstimted = False
        self.usedSelfScheduling = False

        # Commit for self-sceduling functionality pushed on Jan 16 8:20 PM CST 2025
        optimizedTimeStamp = "2025-01-16 08:20 PM"
        opt_datetime = datetime.strptime(optimizedTimeStamp, "%Y-%m-%d %I:%M %p")
        if "ntp640/" == self.logdir:#
            self.trainingEpochs = 1
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/epoch05/" == self.logdir:#in logfile:
            self.trainingEpochs = 5
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/epoch10/" == self.logdir:#in logfile:
            self.trainingEpochs = 10
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/epoch25/" == self.logdir:#in logfile:
            self.trainingEpochs = 25
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif "ndns/epoch50/" == self.logdir:#in logfile:
            self.trainingEpochs = 50
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif "ndns/latest05/" == self.logdir:#in logfile:
            self.trainingEpochs = 5
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/latest10/" == self.logdir:#in logfile:
            self.trainingEpochs = 10
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif "ndns/latest25/" == self.logdir:#in logfile:
            self.trainingEpochs = 25
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif "ndns/latest50/" == self.logdir:#in logfile:
            self.trainingEpochs = 50
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif "ndns/h100_test/" == self.logdir:#in logfile:
            self.trainingEpochs = 0
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/sweep_el/far_right/" == self.logdir:#in logfile:
            self.trainingEpochs = 5
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/sweep_el/midsagittal/" == self.logdir:#in logfile:
            self.trainingEpochs = 50
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif ("ndns/sweep_epochs/cosine_annealing_LR/fixed_orients_1/done/" == self.logdir or #in logfile or
            "ndns/sweep_epochs/cosine_annealing_LR/fixed_orients_2/done/" == self.logdir or #in logfile or
            "ndns/sweep_epochs/cosine_annealing_LR/with_cipic/done/" == self.logdir or #in logfile or 
            "ndns/sweep_epochs/cosine_annealing_LR/without_cipic/" == self.logdir or #in logfile or
            "ndns/sweep_epochs/cosine_annealing_LR/without_cipic/done/" == self.logdir):#in logfile):
            self.trainingEpochs = 5
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif (("ndns/sweep_epochs/default/with_cipic/done/" == self.logdir and "566995_" in logfile) or
              ("ndns/sweep_epochs/default/with_cipic/done/" == self.logdir and "567857_" in logfile)):
            tokens = logfile.split("/")
            logfileName = tokens[-1]
            tokens = logfileName.split("_")
            self.trainingEpochs = int(tokens[-1].replace(".log",""))
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif (("ndns/sweep_epochs/default/with_cipic/done/" == self.logdir and "567859_" in logfile) or
              ("ndns/sweep_epochs/default/with_cipic/done/" == self.logdir and "567860_" in logfile)):
            tokens = logfile.split("/")
            logfileName = tokens[-1]
            tokens = logfileName.split("_")
            self.trainingEpochs = int(tokens[-1].replace(".log",""))
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif (("ndns/sweep_epochs/default/without_cipic/done/"  == self.logdir and "576131_" in logfile) or #in logfile or 
              "ndns/sweep_epochs/multi_step_LR_20_80/with_cipic/done/"  == self.logdir or #in logfile or 
              "ndns/sweep_epochs/multi_step_LR_20_80/without_cipic/done/"  == self.logdir):#in logfile):
#            print("UHHHHH")
#            print(logfile)
            tokens = logfile.split("/")
            logfileName = tokens[-1]
            tokens = logfileName.split("_")
            self.trainingEpochs = int(tokens[-1].replace(".log",""))
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif ("ndns/sweep_epochs/default/without_cipic/done/" == self.logdir and "576132_" in logfile):
#            print("HEEERRRREEE")
#            print(logfile)
            tokens = logfile.split("/")
            logfileName = tokens[-1]
            tokens = logfileName.split("_")
            self.trainingEpochs = int(tokens[-1].replace(".log",""))
            self.validationEpochs = 1
            self.gpuJobLength = 7 * 24 * 60 * 60 
            self.gpuJobLengthDesc = "long"
        elif "ndns/sweep_orients/"  == self.logdir:#in logfile):
            self.trainingEpochs = 20
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/passive_pinna_on_validation_set/"  == self.logdir:#in logfile):
            self.trainingEpochs = 0
            if ("L40_" in logfile):
                self.gpuJobLength = 12 * 60 * 60 
                self.gpuJobLengthDesc = "short"
            elif ("L40M_" in logfile):
                self.gpuJobLength = 24 * 60 * 60 
                self.gpuJobLengthDesc = "medium"
            elif ("L40L_" in logfile):
                self.gpuJobLength = 7 * 24 * 60 * 60 
                self.gpuJobLengthDesc = "long"
            elif ("H100_" in logfile):
                self.gpuJobLength = 7 * 24 * 60 * 60 
                self.gpuJobLengthDesc = "long"
            elif ("H100M_" in logfile):
                self.gpuJobLength = 24 * 60 * 60 
                self.gpuJobLengthDesc = "medium"
        elif ("ndns/pitch_predictor/snn/baseline/"  == self.logdir or 
                "ndns/pitch_predictor/snn/cipic/fixed_orient_1/" == self.logdir):
            self.trainingEpochs = 5
            self.validationEpochs = 1
            self.gpuJobLength = 12 * 60 * 60 
            self.gpuJobLengthDesc = "short"
        elif "ndns/pitch_predictor/snn/baseline/sweeps/width/" == self.logdir:
            self.trainingEpochs = 14
            self.validationEpochs = 1
            self.gpuJobLength = 24 * 60 * 60 
            self.gpuJobLengthDesc = "medium"



#        print(self.gpuJobLengthDesc)
        
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Job submitted from host" in line:
                    self.submitTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                    self.jobID, self.clusterID, self.processID = JobStats.getJobIDFromCHTCLogMessage(line)
                if "Started transferring input files" in line:
                    self.inputFileTransferStartTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                if "Finished transferring input files" in line:
                    self.inputFileTransferEndTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                if "Job executing on host" in line: 
                    self.jobExecutionStartTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                    if self.jobExecutionStartTime > opt_datetime:
                        self.selfSchedulingEnabled = True
                if "Started transferring output files" in line:
                    self.outputFileTransferStartTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                if "Finished transferring output files" in line:
                    self.outputFileTransferEndTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                if "Job terminated." in line or "Job was evicted." in line or "Job was aborted." in line:
                    self.jobTerminationTime = JobStats.getDateTimeFromCHTCLogMessage(line)
                    if "Job was evicted." in line:
                        self.jobWasEvicted = True
                    if "Job was aborted." in line:
                        self.jobWasAborted = True
                if "SlotName:" in line:
                    self.slotName = line.split()[1] 
                    tokens = self.slotName.split("@")
                    self.slotName = tokens[0]
                    self.machineName = tokens[1]
                if "DeviceName" in line:
                    configs = line.split(";")
                    for config in configs:
                        if "DeviceName" in config:
                            assert("=" in config)
                            config =  config.strip()
                            eqLoc = config.index("=")
                            name = config[eqLoc+1:].strip()
                            self.deviceName = name.replace('"', '').strip()
                if "TimeExecute (s)" in line:
                    tokens = line.split(":")
                    self.timeExecuted = int(tokens[1])
                if "TimeSlotBusy (s)" in line:
                    tokens = line.split(":")
                    self.timeSlotBusy = int(tokens[1])
                if "Job terminated of its own accord at " in line and "with exit-code" in line: 
                    tokens = line.split()
                    self.exitCode = int(tokens[-1].strip().replace(".", ""))
                if "disk usage exceeded request_disk" in line:
                    self.diskSpaceExceeded = True
                if "Job failed to complete in" in line:
                    self.jobFailedToComplete = True
                    if "12 hrs" in line:
                        self.gpuJobLength =  12 * 60 * 60 
                        self.gpuJobLengthDesc = "short"
                    elif "24 hrs" in line:
                        self.gpuJobLength =  24 * 60 * 60 
                        self.gpuJobLengthDesc = "medium"
                    else: 
                        print(logfile)
                        print(line)
                if "] Initiating tarball copy" in line:
                    timestamp = JobStats.getDateTimeFromCHTCLogMessage(line)
                    if self.ndnsDatasetTransferStart == None:
                        self.ndnsDatasetTransferStart = timestamp
                    else:
                        if timestamp < self.ndnsDatasetTransferStart:
                            self.ndnsDatasetTransferStart = timestamp
                if "All data has been transferred, starting script" in line:
                    timestamp = JobStats.getDateTimeFromCHTCLogMessage(line)
                    if self.ndnsDatasetTransferEnd == None:
                        self.ndnsDatasetTransferEnd = timestamp
                    else:
                        if timestamp < self.ndnsDatasetTransferEnd:
                            self.ndnsDatasetTransferEnd = timestamp
                if "Completed training and validation" in line:
                    tokens = line.split("epochs_completed:")
                    tokens = tokens[-1]
                    tokens = tokens.split(",")
                    self.trainingEpochs = int(tokens[0].strip())
                    self.validationEpochs = 1
                if "Epoch" in line and " took " in line and "Running avg of epoch latency =" in line and "Time left" in line:
                    if self.gpuJobLength == None:
                        timestamp = JobStats.getDateTimeFromCHTCLogMessage(line)
                        tokens = line.split(". Time left = ")
                        tokens = tokens[-1]
                        tokens = tokens.split()
                        time_left = round(float(tokens[0]))
                        elapsed = (timestamp - self.jobExecutionStartTime).total_seconds()
                        total_time = elapsed + time_left
                        self.gpuJobLengthDesc, self.gpuJobLength = JobStats.roundGpuJobLength(total_time)
                if "Halting training = True" in line:
                    self.usedSelfScheduling = True

        # Check for errors
        if self.timeSlotBusy != None and self.timeSlotBusy > 604800:
            # Jobs can at max run for 7 days
            # try to repair timeslot measurement with other data
            if self.jobTerminationTime != None and self.jobExecutionStartTime != None:
                self.timeSlotBusy = (self.jobTerminationTime - self.jobExecutionStartTime).total_seconds()
                if self.timeSlotBusy > 604800:
                    self.statsCoherent = False
            else:
                self.statsCoherent = False


        testOutFile = logfile.replace(".log", ".out")
        if os.path.exists(testOutFile):
            with open(testOutFile, "r") as f:
                lines = f.readlines()
                resultsStartIdx = 0
                for i in range(len(lines)):
                    line = lines[i]
                    dt_obj = None
                    try: 
                        dt_obj = datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass
                    if dt_obj != None:
                        if self.ndnsDatasetTransferStart == None:
                            self.ndnsDatasetTransferStart = dt_obj
                        else:
                            self.ndnsDatasetTransferEnd = dt_obj
                    if "Detected that this instance in running in a CHTC Job  with" in line:
                        line = line.replace("Detected that this instance in running in a CHTC Job  with ","")
                        line = line.replace(" time remaining.", "")
                        total_seconds = (self.ndnsDatasetTransferEnd  - self.ndnsDatasetTransferStart).total_seconds()
                        if "days, " in line:
                            tokens = line.split("days, ")
                            total_seconds += int(tokens[0]) * 24 * 60 * 60
                            line = tokens[1]
                        t = datetime.strptime(line.strip(), "%H:%M:%S.%f")
                        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                        total_seconds += delta.total_seconds()
                        self.gpuJobLengthDesc, self.gpuJobLength = JobStats.roundGpuJobLength(total_seconds)
                        if 1.0 * self.timeSlotBusy / self.gpuJobLength > 0.80:
                            self.usedSelfScheduling = True
                    if "Subject, Channel, Speech Orient, Noise Orient," in line and "Final Validation Score SI-SNR (dB)" in line:
                        resultsStartIdx = i
                if "ndns/passive_pinna_on_validation_set/"  == self.logdir:
                    self.validationEpochs = len(lines) - resultsStartIdx
#                    if "ndns/passive_pinna_on_validation_set/L40M_003_0_30_0_300_1308673.log" == logfile:
#                        print("HEEEEEEEEEERRRREEEEEE: " + str(self.validationEpochs))



        # Final attempt to guess gpuJobLength
        if self.gpuJobLength == None and self.timeSlotBusy != None:
            self.gpuJobLengthEstimted = True
            self.gpuJobLengthDesc, self.gpuJobLength = JobStats.roundGpuJobLength(self.timeSlotBusy)

    def __hash__(self):
        return hash(self.jobID)

    def __eq__(self, other):
        if not isinstance(other, JobStats):
            return False
        return self.jobID == other.jobID

    def __str__(self):
        return self.jobID

    @staticmethod
    def wereColocated(job1, job2):
        if not isinstance(job1, JobStats):
            return False
        if not isinstance(job2, JobStats):
            return False
        if job1.machineName != job2.machineName:
            return False

        if job1.jobExecutionStartTime == None:
            return False
        if job1.jobTerminationTime == None:
            return False
        if job2.jobExecutionStartTime == None:
            return False
        if job2.jobTerminationTime == None:
            return False
        
        if job1.jobExecutionStartTime <= job2.jobExecutionStartTime and job2.jobTerminationTime <= job1.jobTerminationTime:
            return True

        if job2.jobExecutionStartTime <= job1.jobExecutionStartTime and job1.jobTerminationTime <= job2.jobTerminationTime:
            return True

        return False

    @staticmethod
    def roundGpuJobLength(seconds):
        if seconds > 3 * 24 * 60 * 60:
            return "long", 7*24*60*60
        elif seconds > 24 * 60 * 60:
            return "mediumlong", 3*24*60*60
        elif seconds > 12 * 60 * 60:
            return "medium", 24*60*60
        else:
            return "short", 12*60*60

    @staticmethod
    def getDateTimeFromCHTCLogMessage(msg):
        tokens = msg.split()
        date_string = tokens[2] + " " + tokens[3]
        date_format = "%Y-%m-%d %H:%M:%S"
        datetime_object = datetime.strptime(date_string, date_format)
        return datetime_object

    @staticmethod
    def getJobIDFromCHTCLogMessage(msg):
        tokens = msg.split()
        jobID = tokens[1].replace("(", "").replace(")", "")
        tokens = jobID.split(".")
        clusterID = tokens[0]
        processID = tokens[1] + "." + tokens[2]
        return jobID, clusterID, processID

    def getLogFile(self):
        return self.logfile

    def getSubmitTime(self):
        return self.submitTime

    def getInputFileTransferStartTime(self):
        return self.inputFileTransferStartTime

    def getInputFileTransferEndTime(self):
        return self.inputFileTransferEndTime

    def getOutputFileTransferStartTime(self):
        return self.outputFileTransferStartTime

    def getOutputFileTransferEndTime(self):
        return self.outputFileTransferEndTime

    def getExecutionStartTime(self):
        return self.jobExecutionStartTime

    def getJobTerminationTime(self):
        return self.jobTerminationTime

    def getSlotName(self):
        return self.slotName

    def getMachineName(self):
        return self.machineName

    def getTrainingEpochs(self):
        return self.trainingEpochs

    def getValidationEpochs(self):
        return self.validationEpochs

    def getDeviceName(self):
        return self.deviceName

    def getGpuJobLength(self):
        return self.gpuJobLength

    def getGpuJobLengthDesc(self):
        return self.gpuJobLengthDesc

    def getTimeExecuted(self):
        return self.timeExecuted

    def getTimeSlotBusy(self):
        return self.timeSlotBusy

    def getNDNSDatasetTransferStart(self):
        return self.ndnsDatasetTransferStart

    def getNDNSDatasetTransferEnd(self):
        return self.ndnsDatasetTransferEnd

    def getExitCode(self):
        return self.exitCode

    def exitedSuccessfully(self):
        return self.exitCode == 0

    def statsAreCoherent(self):
        return self.statsCoherent

    def wasJobEvisted(self):
        return self.jobWasEvicted

    def didJobExceedDiskSpace(self):
        return self.diskSpaceExceeded

    def didJobCompleteInTime(self):
        return not self.jobFailedToComplete

    def didJobAbort(self):
        return self.jobWasAborted

    def wasSelfScheduledEnabled(self):
        return self.selfSchedulingEnabled
    
    def wasSelfSchedulingUsed(self):
        return self.selfSchedulingEnabled and self.usedSelfScheduling

    def wasGpuJobLengthEstimated(self):
        return self.gpuJobLengthEstimted

class UniqueJobPair:

    def __init__(self, job1, job2):
        if not isinstance(job1, JobStats):
            assert(False)
        if not isinstance(job2, JobStats):
            assert(False)
        self.job1 = job1
        self.job2 = job2

    def __hash__(self):
        return hash(self.job1.__hash__()) + hash(self.job2.__hash__())

    def __eq__(self, other):
        if not isinstance(other, UniqueJobPair):
            return False
        if self.job1 == other.job1 and self.job2 == other.job2:
            return True
        if self.job1 == other.job2 and self.job2 == other.job1:
            return True
        return False

    def __str__(self):
        string = "(" + str(self.job1) + "," + str(self.job2) + ")"
        return string


def get_log_files(mypath):
    onlyfiles = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f)) and ".log" in f]
    return onlyfiles

logfiles = []
for srcDir in directories:
    logfiles += get_log_files(srcDir)

allJobs = set()
shortJobs = set()
mediumJobs = set()
mediumLongJobs = set()
longJobs = set()
for logfile in logfiles:
    job = JobStats(logfile)
    # Skip jobs that crashed
    if not job.exitedSuccessfully():
        continue
    # Skip jobs that have stats that don't make sense
    if not job.statsAreCoherent():
        continue
    # Skip jobs that did not complete under time limit
    if not job.didJobCompleteInTime():
        continue
    # Skip jobs that crashed due to disk space
    if job.didJobExceedDiskSpace():
        continue
    # Skip jobs that were manually rm'ed 
    if job.didJobAbort():
        continue
    # Skip if it is optimized job
    if job.wasSelfSchedulingUsed():
        continue
    allJobs.add(job)
    if job.getGpuJobLengthDesc() == "short":
        shortJobs.add(job)
    if job.getGpuJobLengthDesc() == "medium":
        mediumJobs.add(job)
    if job.getGpuJobLengthDesc() == "mediumlong":
        mediumLongJobs.add(job)
    if job.getGpuJobLengthDesc() == "long":
        longJobs.add(job)

optAllJobs = set()
optShortJobs = set()
optMediumJobs = set()
optMediumLongJobs = set()
optLongJobs = set()
for logfile in logfiles:
    job = JobStats(logfile)
    # Skip jobs that crashed
    if not job.exitedSuccessfully():
        continue
    # Skip jobs that have stats that don't make sense
    if not job.statsAreCoherent():
        continue
    # Skip jobs that did not complete under time limit
    if not job.didJobCompleteInTime():
        continue
    # Skip jobs that crashed due to disk space
    if job.didJobExceedDiskSpace():
        continue
    # Skip jobs that were manually rm'ed 
    if job.didJobAbort():
        continue
    # skip if it's non-optimized job
    if not job.wasSelfSchedulingUsed():
        continue
#    if job.getDeviceName() == "NVIDIA L40":
#        print(job.getLogFile())
    optAllJobs.add(job)
    if job.getGpuJobLengthDesc() == "short":
        optShortJobs.add(job)
    if job.getGpuJobLengthDesc() == "medium":
        optMediumJobs.add(job)
    if job.getGpuJobLengthDesc() == "mediumlong":
        optMediumLongJobs.add(job)
    if job.getGpuJobLengthDesc() == "long":
        optLongJobs.add(job)
print("Read info from " + str(len(allJobs)) + " jobs.")
print("Detected self scheduling opt enabled in " + str(len(optAllJobs)) + " jobs")
shortJobsRuntimeExceeded = set()
mediumJobsRuntimeExceeded = set()
mediumLongJobsRuntimeExceeded = set()
longJobsRuntimeExceeded = set()
allJobsRuntimeExceeded = set()
for logfile in logfiles:
    job = JobStats(logfile)
    # Skip jobs that crashed due to disk space
    if job.didJobExceedDiskSpace():
        continue
    # skip if it's optimized job
    if job.wasSelfSchedulingUsed():
        continue
    if not job.didJobCompleteInTime():
        allJobsRuntimeExceeded.add(job)
        if job.getGpuJobLengthDesc() == "short":
            shortJobsRuntimeExceeded.add(job)
        if job.getGpuJobLengthDesc() == "medium":
            mediumJobsRuntimeExceeded.add(job)
        if job.getGpuJobLengthDesc() == "mediumlong":
            mediumLongRuntimeExceededJobs.add(job)
        if job.getGpuJobLengthDesc() == "long":
            longJobsRuntimeExceeded.add(job)

optShortJobsRuntimeExceeded = set()
optMediumJobsRuntimeExceeded = set()
optMediumLongJobsRuntimeExceeded = set()
optLongJobsRuntimeExceeded = set()
optAllJobsRuntimeExceeded = set()
for logfile in logfiles:
    job = JobStats(logfile)
    # Skip jobs that crashed due to disk space
    if job.didJobExceedDiskSpace():
        continue
    # skip if it's non-optimized job
    if not job.wasSelfSchedulingUsed():
        continue
    if not job.didJobCompleteInTime():
        optAllJobsRuntimeExceeded.add(job)
        if job.getGpuJobLengthDesc() == "short":
            optShortJobsRuntimeExceeded.add(job)
        if job.getGpuJobLengthDesc() == "medium":
            optMediumJobsRuntimeExceeded.add(job)
        if job.getGpuJobLengthDesc() == "mediumlong":
            optMediumLongRuntimeExceededJobs.add(job)
        if job.getGpuJobLengthDesc() == "long":
            optLongJobsRuntimeExceeded.add(job)

#### Now start computing all the statistics

def printStat(stat, data):
    if len(data) != 0:
        string = stat + "," 
        string += str(min(data)) + ","
        string += str(sum(data)/len(data)) + "," 
        string += str(max(data)) + ","
        string += str(len(data))
    else:
        string = stat + "," 
        string += "0,"
        string += "0," 
        string += "0,"
        string += str(len(data))
    print(string)

def getYieldTimes(jobSubset):
    times = []
    for job in jobSubset:
        used = job.getTimeSlotBusy()
        total = job.getGpuJobLength()
        if used == None or total == None:
            continue
        used = int(used)
        total = int(total)
        if used > total:
            print(job.getLogFile())
            print("Used: " + str(used))
            print("Total: " + str(total))
        times.append(total - used)
    return times

def getIdleTimes(jobSubset):
    times = []
    for job in jobSubset:
        start = job.getSubmitTime()
        end = job.getExecutionStartTime()
        if start == None or end == None:
            continue
        diff = end - start
        times.append(diff.total_seconds())
    return times

def getExecTimes(jobSubset):
    executionTimes = []
    for job in jobSubset:
        start = job.getExecutionStartTime()
        end = job.getJobTerminationTime()
        diff = end - start
        executionTimes.append(diff.total_seconds())
    return executionTimes

def getDatasetTransferTimes(jobSubset):
    times = []
    for job in jobSubset:
        start = job.getNDNSDatasetTransferStart()
        end =  job.getNDNSDatasetTransferEnd()
        if start == None or end == None:
            continue
        diff = end - start
        times.append(diff.total_seconds())
    return times

print()
#print("Short Idle Times:")
#for time in getIdleTimes(shortJobs.union(optShortJobs)):
#    print(time)
#print("Medium Idle Times:")
#for time in getIdleTimes(mediumJobs.union(optMediumJobs)):
#    print(time)
#print("Long Idle Times:")
#for time in getIdleTimes(longJobs.union(optLongJobs)):
#    print(time)

print("Short Dataset Transfer Times:--------------------------")
for time in getDatasetTransferTimes(shortJobs.union(optShortJobs)):
    print(time)
print("Medium Dataset Transfer Times:")
for time in getDatasetTransferTimes(mediumJobs.union(optMediumJobs)):
    print(time)
print("MediumLong Dataset Transfer Times:")
for time in getDatasetTransferTimes(mediumLongJobs.union(optMediumLongJobs)):
    print(time)
print("Long Dataset Transer Times:")
for time in getDatasetTransferTimes(longJobs.union(optLongJobs)):
    print(time)

print("Unoptimized Jobs")

print("GPUJobLength,Stat,Mean,Max,Min,Count")
printStat("Short,IdleTime", getIdleTimes(shortJobs))
printStat("Medium,IdleTime", getIdleTimes(mediumJobs))
printStat("MediumLong,IdleTime", getIdleTimes(mediumLongJobs))
printStat("Long,IdleTime", getIdleTimes(longJobs))
printStat("All,IdleTime", getIdleTimes(allJobs))
printStat("Short,DatasetTransferTime", getDatasetTransferTimes(shortJobs))
printStat("Medium,DatasetTransferTime", getDatasetTransferTimes(mediumJobs))
printStat("MediumLong,DatasetTransferTime", getDatasetTransferTimes(mediumLongJobs))
printStat("Long,DatasetTransferTime", getDatasetTransferTimes(longJobs))
printStat("All,DatasetTransferTime", getDatasetTransferTimes(allJobs))
printStat("Short,ExecutionTime", getExecTimes(shortJobs))
printStat("Medium,ExecutionTime", getExecTimes(mediumJobs))
printStat("MediumLong,ExecutionTime", getExecTimes(mediumLongJobs))
printStat("Long,ExecutionTime", getExecTimes(longJobs))
printStat("All,ExecutionTime", getExecTimes(allJobs))
printStat("Short,YieldTime", getYieldTimes(shortJobs))
printStat("Medium,YieldTime", getYieldTimes(mediumJobs))
printStat("MediumLong,YieldTime", getYieldTimes(mediumLongJobs))
printStat("Long,YieldTime", getYieldTimes(longJobs))
printStat("All,YieldTime", getYieldTimes(allJobs))
print()
print("Short jobs that exceeded runtime: " + str(len(shortJobsRuntimeExceeded)))
print("Medium jobs that exceeded runtime: " + str(len(mediumJobsRuntimeExceeded)))
print("MediumLong jobs that exceeded runtime: " + str(len(mediumLongJobsRuntimeExceeded)))
print("Long jobs that exceeded runtime: " + str(len(longJobsRuntimeExceeded)))
print()
print("---------------------------------------------------------------------")
print("Optimized Jobs")
print("GPUJobLength,Stat,Mean,Max,Min,Count")
printStat("Short,IdleTime", getIdleTimes(optShortJobs))
printStat("Medium,IdleTime", getIdleTimes(optMediumJobs))
printStat("MediumLong,IdleTime", getIdleTimes(optMediumLongJobs))
printStat("Long,IdleTime", getIdleTimes(optLongJobs))
#printStat("All,IdleTime", getIdleTimes(optAllJobs))
printStat("Short,DatasetTransferTime", getDatasetTransferTimes(optShortJobs))
printStat("Medium,DatasetTransferTime", getDatasetTransferTimes(optMediumJobs))
printStat("MediumLong,DatasetTransferTime", getDatasetTransferTimes(optMediumLongJobs))
printStat("Long,DatasetTransferTime", getDatasetTransferTimes(optLongJobs))
printStat("All,DatasetTransferTime", getDatasetTransferTimes(optAllJobs))
printStat("Short,ExecutionTime", getExecTimes(optShortJobs))
printStat("Medium,ExecutionTime", getExecTimes(optMediumJobs))
printStat("MediumLong,ExecutionTime", getExecTimes(optMediumLongJobs))
printStat("Long,ExecutionTime", getExecTimes(optLongJobs))
printStat("All,ExecutionTime", getExecTimes(optAllJobs))
printStat("Short,YieldTime", getYieldTimes(optShortJobs))
printStat("Medium,YieldTime", getYieldTimes(optMediumJobs))
printStat("MediumLong,YieldTime", getYieldTimes(optMediumLongJobs))
printStat("Long,YieldTime", getYieldTimes(optLongJobs))
printStat("All,YieldTime", getYieldTimes(optAllJobs))
print()
print("Short jobs that exceeded runtime: " + str(len(optShortJobsRuntimeExceeded)))
print("Medium jobs that exceeded runtime: " + str(len(optMediumJobsRuntimeExceeded)))
print("MediumLong jobs that exceeded runtime: " + str(len(optMediumLongJobsRuntimeExceeded)))
print("Long jobs that exceeded runtime: " + str(len(optLongJobsRuntimeExceeded)))
print()

# Determine number of colocated jobs
colocatedJobPairs = set()
for job1 in allJobs.union(allJobsRuntimeExceeded).union(optAllJobs).union(optAllJobsRuntimeExceeded):
    for job2 in allJobs.union(allJobsRuntimeExceeded).union(optAllJobs).union(optAllJobsRuntimeExceeded):
        if job1 != job2 and JobStats.wereColocated(job1, job2):
            colocatedJobPairs.add(UniqueJobPair(job1, job2))
print("Found " + str(len(colocatedJobPairs)) +  " jobs that were colocated of total " + str(len(allJobs.union(allJobsRuntimeExceeded).union(optAllJobs).union(optAllJobsRuntimeExceeded))))

print("Unoptimized")
latencies = dict()
for job in allJobs.union(allJobsRuntimeExceeded):
    dsTxStart = job.getNDNSDatasetTransferStart()
    dsTxEnd =  job.getNDNSDatasetTransferEnd()
    total_time =job.getTimeSlotBusy() 
    tEpochs = job.getTrainingEpochs()
    vEpochs = job.getValidationEpochs()
    deviceName = job.getDeviceName()
    if (dsTxStart != None and dsTxEnd != None and total_time != None and tEpochs !=None and vEpochs != None):
        delta = dsTxEnd - dsTxStart
        timePerEpoch = (float(total_time) - delta.total_seconds()) / float(tEpochs + vEpochs)
        if deviceName in latencies.keys():
            latencies[deviceName].append(timePerEpoch)
        else:
            latencies[deviceName] = [timePerEpoch]


for k,v in latencies.items():
    print(k + "," + str(sum(v)/len(v)))

print("Optimized")
optLatencies = dict()
for job in optAllJobs:
    dsTxStart = job.getNDNSDatasetTransferStart()
    dsTxEnd =  job.getNDNSDatasetTransferEnd()
    total_time =job.getTimeSlotBusy() 
    tEpochs = job.getTrainingEpochs()
    vEpochs = job.getValidationEpochs()
    deviceName = job.getDeviceName()
    if (dsTxStart != None and dsTxEnd != None and total_time != None and tEpochs !=None and vEpochs != None):
        if tEpochs > 0:
            continue
        delta = dsTxEnd - dsTxStart
#        if (deviceName == "NVIDIA A100-SXM4-80GB"): # "NVIDIA L40"):
#            print(job.getLogFile())
#            print("Time slot Time: " + str(total_time))
#            print("Dataset Tx Time: " + str(delta.total_seconds()))
#            print("tEpochs: " + str(tEpochs))
#            print("vEpochs: " + str(vEpochs))
        timePerEpoch = (float(total_time) - delta.total_seconds()) / float(tEpochs + vEpochs)
        if deviceName in optLatencies.keys():
            optLatencies[deviceName].append(timePerEpoch)
        else:
            optLatencies[deviceName] = [timePerEpoch]


for k,v in optLatencies.items():
    print(k + "," + str(sum(v)/len(v)))
#    if (k == "NVIDIA A100-SXM4-80GB"): # "NVIDIA L40"):
#    if (k == "NVIDIA L40"):
#        print(v)

