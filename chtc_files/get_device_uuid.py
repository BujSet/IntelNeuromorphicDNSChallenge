import os

UUID = ""
with open("../.update.ad", "r") as f:
    lines = f.readlines()
    for line in lines:
        if ("GPUs_DeviceUuid" in line):
            tokens = line.split("=")
            uuid = tokens[1].replace('"', '').strip()
            UUID = "GPU-" + uuid
print(UUID)            
