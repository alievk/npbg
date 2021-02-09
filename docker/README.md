### Docker image for Neural Point-Based Graphics

This image can be helpful in two major cases:
* for smooth preparation of the training/inference environment,
* for running NPBG in headless mode (when no GPU-capable machines with X server are available).

The Dockerfile is prepared for a machine with at least 10.2 CUDA capability. In case the CUDA capability of the machine is lower, one can try changing the version of CUDA in the image name (line 1 of Dockerfile) and the version of cudatoolkit (line 57 of Dockerfile).

To build the container, go to the repository root folder, specify directory of the data files (to be mounted in the container) in constants.sh (line 5) and then run:

```bash
cd docker/tools
bash build.sh
```

To run the training job:
```bash
cd docker/tools
GPU_NUMBER=0              # GPU number to use (use 0 if only one GPU is available)
SAVE_DIR=/path/to/logs    # logs folder which must be inside the $DATA_DIR specified in constants.sh    
bash run_x.sh ${GPU_NUMBER} python train.py --config configs/train_example.yaml --pipeline npbg.pipelines.ogl.TexturePipeline --save_dir ${SAVE_DIR} ...
```

where `...` can be replaced with additional training parameters, if needed.

Note that this image is only meant for running training and inference, but not for the interactive viewer (in case the machine has no X server). One can train the network and descriptors on the remote server and then download the learned tensors to the machine with the X server and run the interactive viewer.