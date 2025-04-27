from htcondor.htchirp import HTChirp
import os, time, datetime, sys

def send_log_msg(message):
    with HTChirp() as chirp:
        chirp.ulog(message)

def get_job_start():
    with HTChirp() as chirp:
        return float(chirp.get_job_attr("JobCurrentStartDate"))

def get_gpu_job_length():
    with HTChirp() as chirp:
        lengthStr = chirp.get_job_attr("GPUJobLength")
        if "short" in lengthStr:
            # 12 hour job length
            return 12*60*60
        elif "mediumlong" in lengthStr:
            # 3 day job length
            return 3*24*60*60
        elif "medium" in lengthStr:
            # 1 day job length
            return 24*60*60
        elif "long" in lengthStr:
            # 7 day job length
            return 7*24*60*60
        else:
            print("Unable to parse GPU job length:" + lengthStr)
            sys.exit(1)

def get_gpu_time_remaining(rawValue=False):
    elapsed = time.time() - get_job_start()
    remaining = get_gpu_job_length() - elapsed
    if rawValue == True:
        return remaining
    return str(datetime.timedelta(seconds=remaining))

def get_cpu_job_length():
    with HTChirp() as chirp:
        return float(chirp.get_job_attr("AllowedExecuteDuration"))

def get_cpu_time_remaining(rawValue=False):
    elapsed = time.time() - get_job_start()
    remaining = get_cpu_job_length() - elapsed
    if rawValue == True:
        return remaining
    return str(datetime.timedelta(seconds=remaining))
