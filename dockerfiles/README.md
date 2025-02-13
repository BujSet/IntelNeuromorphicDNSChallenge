## Dockerfile
These dockerfiles describe the image that contains all the dependences 
necessary to run the Intel NDNS challenge baseline model on an NVIDIA GPU. 
Also includes necessary dependencies for working with the CIPIC database. 
Dockerfiles are now user specific. This was done to allow the pytorch profiler 
to collect data when jobs run on CHTC. If you don't care about profiling data,
and dockerfile can be used.

### Notes

* Building the image is not strictly necessary, as the image can be pulled from Dockerhub via:

```
docker pull rselagam/ndns:v52_cuda12.4.0_selagamsetty
```

* Please be aware that the image requires ~ 40 GB of disk space.

### Building the image

However, should you wish to build the image itself, you can run the following:

```
docker build -t <image_name> .
```
