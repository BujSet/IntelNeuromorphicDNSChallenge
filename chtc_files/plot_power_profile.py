import matplotlib.pyplot as plt
from datetime import datetime

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"

def strptime(val):
    if '.' not in val:
        return datetime.strptime(val, TIMESTAMP_FORMAT)

    nofrag, frag = val.split(".")
    date = datetime.strptime(nofrag, TIMESTAMP_FORMAT)

    frag = frag[:6]  # truncate to microseconds
    frag += (6 - len(frag)) * '0'  # add 0s
    return date.replace(microsecond=int(frag))

def collect(powerFile):
    x = []
    y = []
    with open(powerFile, "r") as f:
        lines = f.readlines()
        lines = lines[1:] # get rid of headers
        values = lines[0].split(",")
        t_start = strptime(values[0])
        for line in lines:
            values = line.split(",")
            time = strptime(values[0])
            power = float(values[1].replace("W", "").strip())
            delta = time - t_start
            x.append(delta.total_seconds())
            y.append(power)
    return x, y

GPU_Name = ""
with open("../.update.ad", "r") as f:
    lines = f.readlines()
    for line in lines:
        if ("GPUs_DeviceName" in line):
            tokens = line.split("=")
            GPU_Name = tokens[1].replace('"', '').strip()

def calcEnergy(x, y):
    energy = 0.0
    for i in range(len(x)-1):
        dt = x[i+1] - x[i]
        de = y[i] * dt
        energy += de
    return energy

x, y = collect("../power.txt")
energy = round(calcEnergy(x,y) / 1000.0, 3)
titleString = GPU_Name + "\n" + str(energy) + " kJ"
#print(x)
#print(y)
#print(energy)
#print(titleString)
plt.plot(x, y)
plt.xlabel("Time (sec)")
plt.ylabel("Dynamic Power (W)")
plt.title(titleString)
plt.savefig("power_profile.png", bbox_inches="tight")

