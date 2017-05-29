FROM tensorflow/tensorflow:1.2.0-rc0-gpu

MAINTAINER Sanghoon Yoon <shygiants@gmail.com>

# Copy all source codes
COPY . /rumor

# Set working directory
WORKDIR "/rumor"

RUN mkdir /summary
RUN mkdir /dataset

ENTRYPOINT ["bash", "run_tensorflow.sh"]