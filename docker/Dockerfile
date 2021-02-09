FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

## Base packages for ubuntu

# clean the libs list
RUN apt-get clean
RUN apt-get update -qq
RUN apt-get install -y \
    git \
    wget \
    bzip2 \
    htop \
    vim \
    nano \
    g++ \
    make \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    curl \
    libglew-dev \
    libglfw3-dev \
    libjpeg-dev \
    libjsoncpp-dev \
    libpng-dev \
    lxde \
    mesa-utils \
    x11vnc \
    xorg-dev \
    xvfb \
    ne


## Download and install miniconda

RUN wget https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O /tmp/miniconda.sh

RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

## Install base conda packages

RUN conda install -y numpy scipy pandas matplotlib pillow scikit-learn 

RUN conda list

## Torch
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

## OpenCV

RUN conda install -c conda-forge opencv

## Additional packages

RUN pip install tqdm \
                h5py \
                scikit-image \
                einops \
                graphviz \
                plyfile \
                munch \
                huepy \
                Cython

RUN pip install \
                tensorflow \
                tensorboardX \
                pytorch-memlab \
                albumentations
 
RUN pip install \
                PyOpenGL \
                PyOpenGL_accelerate \
                trimesh
                
# RUN pip install glumpy numpy-quaternion
RUN pip install git+https://github.com/DmitryUlyanov/glumpy \
                numpy-quaternion

#RUN wget https://files.pythonhosted.org/packages/9c/1d/4544708aaa89f26c97cc09450bb333a23724a320923e74d73e028b3560f9/PyOpenGL-3.1.0.tar.gz && \
#    tar -zxvf PyOpenGL-3.1.0.tar.gz && \
#    cd PyOpenGL-3.1.0 && \
#    python setup.py install && \
#    cd ..


# Add non-root user
ARG USERNAME=docker
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml


WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
USER $USERNAME:$USERNAME

COPY --chown=docker ./setup/.bashrc .
RUN chmod a+x .bashrc

ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
