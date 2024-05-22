## NDNS Dockerfile
This dockerfile describe the image that contains all the dependences necessary to run the Intel NDNS challenge baseline model on an NVIDIA GPU. Also includes necessary dependencies for working with the CIPIC database.

### Notes

* Building the image is not strictly necessary, as the image can be pulled from Dockerhub via:

```
docker pull rselagam/ndns:v47_cuda12.1.1
```

* Please be aware that the image requires ~ 20 GB of disk space.

### Building the image

However, should you wish to build the image itself, you can run the following:

```
docker build -t <image_name> .
```
