import os 
import time
import datetime

class JobAd(object):
    def __init__(self, dir_path):
        file_path = os.path.join(dir_path, ".job.ad")
        if not os.path.exists(file_path):
            self.jobReadSuccess = False
            print("Error, unable to find .job.ad file in HOME directory")
            return

        self.attributes = dict()
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(" = ")
                if (len(tokens) != 2):
                    print(tokens)
                assert(len(tokens) == 2)
                key = tokens[0].strip()
                value = tokens[1].strip()
                try:
                    value = int(value)
                except:
                    pass
                if (key == "GPUJobLength"):
                    if value == '"short"':
                        value = 12*60*60
                    elif (value == '"medium"'):
                        value = 24*60*60
                    elif (value == '"long"'):
                        value = 7*24*60*60
                self.attributes[key] = value
        self.jobReadSuccess = True
        print("Read " + str(len(self.attributes.keys())) + " job attributes")

    def get_job_start_time(self):
        if not self.jobReadSuccess:
            return -1
        return self.attributes['JobCurrentStartDate']

    def get_job_elapsed_runtime(self):
        if not self.jobReadSuccess:
            return -1
        start =  self.attributes['JobCurrentStartDate']
        return time.time() - start

    def get_gpu_job_time_remaining(self, rawValue=False):
        if not self.jobReadSuccess:
            return -1
        elapsed = time.time() - self.attributes['JobCurrentStartDate']
        remaining = self.attributes["GPUJobLength"] - elapsed
        if rawValue == True:
            return remaining
        return str(datetime.timedelta(seconds=remaining))

CurrentJob = JobAd(os.environ.get("HOME"))
