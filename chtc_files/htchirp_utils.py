from htcondor.htchirp import HTChirp
import os, time, datetime, sys

def send_log_msg(message):
    with HTChirp() as chirp:
        chirp.ulog(message)

def get_job_start():
    with HTChirp() as chirp:
        datestr = chirp.get_job_attr("JobCurrentStartDate")
        print(datestr)
        dateobj = datetime.strptime(datestr, "%b %d %Y %I:%Mp")
        return dateobj.time()

def get_gpu_job_length():
    with HTChirp() as chirp:
        lengthStr = chirp.get_job_attr("GPUJobLength")
        if lengthStr == '"short"':
            return 12*60*60
        elif lengthStr == '"medium"':
            return 24*60*60
        elif lengthStr == '"long"':
            return 7*24*60*60
        else:
            print("Unable to parse GPU job length:" + lengthStr)
            sys.exit(1)

def get_gpu_time_remaining(rawValue=False):
    elapsed = time.time() - get_job_start()
    print(elapsed)
    remaining = get_gpu_job_length() - elapsed
    if rawValue == True:
        return remaining
    return str(datetime.timedelta(seconds=remaining))
