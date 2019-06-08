# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /app
WORKDIR /

# Copy the current directory contents into the container at /
COPY . /

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc
RUN apt-get install -y g++
RUN apt-get install unzip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get update && apt-get install make
RUN unzip /trec_eval.zip
RUN cd /trec_eval; make
RUN chmod +x /trec_eval/*
RUN chmod +x /index
RUN chmod +x /train
RUN chmod +x /init
RUN chmod +x /search
RUN chmod +x /extract_robust.sh

