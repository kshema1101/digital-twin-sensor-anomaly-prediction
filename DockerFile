# python image
# FROM python:3.11.3-slim 
FROM nvcr.io/nvidia/tensorrt:24.02-py3
#set the working directory
WORKDIR /Anomaly_prediction
#copy the current directory contents into the container at /app
COPY . /Anomaly_prediction
#install dependencies
RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache-dir -r requirement.txt
#command to run on container start
CMD ["python", "data_generation/pythonbased/synthetic_data.py"]


# # Start with the NVIDIA TensorRT image
# FROM nvcr.io/nvidia/tensorrt:24.02-py3

# # Set the working directory
# WORKDIR /Anomaly_prediction

# # Copy the files
# COPY . /Anomaly_prediction

# # Install GDAL and other dependencies
# RUN apt-get update && apt-get install -y \
#     gdal-bin \
#     libgdal-dev \
#     && apt-get clean

# # Set GDAL config path
# ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
# ENV C_INCLUDE_PATH=/usr/include/gdal
# ENV GDAL_VERSION=3.4.1  
# #Adjust this version based on your GDAL requirement

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirement.txt

