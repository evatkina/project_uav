#FROM nvidia/cuda:12.3.1-base-ubuntu20.04
FROM ubuntu:22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/*

## create project folder and copy code into it
RUN mkdir -p /app
# Mount the current directory contents into the container at /app
ADD Codes /app
ADD environment.yaml /app
# Change working dir to /app
WORKDIR /app


# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH


# Install Python dependencies
RUN micromamba create -y -n base -f environment.yaml

# Set folder to mount video files
RUN mkdir -p /app/testvideo

EXPOSE 8501
EXPOSE 8000

CMD ["streamlit", "run", "app.py"]
