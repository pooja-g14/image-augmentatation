# Start from UBUNTU image
FROM ubuntu:22.04
# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends git wget
# Download files to user config dir
RUN mkdir -p /root/.config/Ultralytics/ && \
    wget --no-check-certificate -P /root/.config/Ultralytics/ https://ultralytics.com/assets/Arial.ttf && \
    wget --no-check-certificate -P /root/.config/Ultralytics/ https://ultralytics.com/assets/Arial.Unicode.ttf
# Create working directory
RUN mkdir -p /image-augmentation
WORKDIR /image-augmentation
# Disable SSL verification (if necessary)
RUN git config --global http.sslVerify false
# Clone the repository
RUN git clone https://github.com/pooja-g14/image-augmentation.git .
