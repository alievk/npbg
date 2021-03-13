# Neural Point-Based Graphics

### <img align=center src=./docs/images/project.png width='32'/> [Project](https://saic-violet.github.io/npbg) &ensp; <img align=center src=./docs/images/video.png width='24'/> [Video](https://youtu.be/2uIe4iD4gSY) &ensp; <img align=center src=./docs/images/paper.png width='24'/> [Paper](https://arxiv.org/abs/1906.08240v3)

[Neural Point-Based Graphics](https://arxiv.org/abs/1906.08240v3)<br>
[Kara-Ali Aliev](https://github.com/alievk)<sup>1</sup> &nbsp;
[Artem Sevastopolsky](https://seva100.github.io)<sup>1,2</sup> &nbsp;
[Maria Kolos](https://github.com/mvkolos)<sup>1,2</sup> &nbsp;
[Dmitry Ulyanov](https://dmitryulyanov.github.io/about)<sup>3</sup> &nbsp;
[Victor Lempitsky](http://sites.skoltech.ru/compvision/members/vilem/)<sup>1,2</sup> <br>
<sup>1</sup>Samsung AI Center &nbsp; <sup>1</sup>Skolkovo Institute of Science and Technology &nbsp; <sup>3</sup>in3d.io

<img src=docs/images/teaser.jpg width=1200>

**UPD (09.02.2021)**: added a Docker container which can be executed on a headless node. See [Docker Readme](https://github.com/alievk/npbg/tree/master/docker).

## About

This is PyTorch implementation of Neural Point-Based Graphics (NPBG), a new method for realtime photo-realistic rendering of real scenes. NPBG uses a raw point cloud as the geometric representation of a scene, and augments each point with a learnable neural descriptor that encodes local geometry and appearance. A deep rendering network is learned in parallel with the descriptors, so that new views of the scene can be obtained by passing the rasterizations of a point cloud from new viewpoints through this network.

<img src=docs/images/pipeline.jpg width=1200>

## Setup

The following instructions describe installation of conda environment. If you wish to setup the Docker environment, see the Readme in the [docker folder](https://github.com/alievk/npbg/tree/master/docker). This way is also recommended for headless machines (without X server enabled).

Run this command to install python environment using [conda](https://docs.conda.io/en/latest/miniconda.html):
```bash
source scripts/install_deps.sh
```

## Run

You can render one of the fitted scenes we provide right away in the real-time viewer or fit your own scene.

Download fitted scenes and universal rendering network weights from [here](https://drive.google.com/open?id=1h0VrTBRyNKUCk3y5GnxhzdA7C6_4ok1Z) and unpack in the sources root directory.

We suppose that you have at least one GeForce GTX 1080 Ti for fitting and inference.

### Viewer navigation:

* Rotation: press left mouse button and drag
* Move: press rigth mouse button and drug / scroll middle mouse botton
* Pan: press middle mouse button and drug

### Use fitted scene

Here we show a couple of examples how to run fitted scenes in the viewer.

#### Person 1
```bash
python viewer.py --config downloads/person_1.yaml --viewport 2000,1328 --origin-view
```
Since this scene was fitted on 4k images, we crop image size with `--viewport` argument to fit the scene into memory of a modest GPU.

#### Studio
```bash
python viewer.py --config downloads/studio.yaml --rmode fly
```

Check `downloads` directory for more examples.

### Fit your scene

Fitting a new scene consists of two steps:

1. Point cloud reconstruction
2. Fitting descriptors

There is a bunch of software for point cloud reconstruction. While it is possible to adopt different software packages for our pipeline, we will choose Agisoft Metashape for this demonstration.

#### Point cloud reconstruction (Agisoft Metashape)

If you don't have a license for Agisoft Metashape Pro, start a trial version by filling in [the form](https://www.agisoft.com/downloads/request-trial/). On the first start, enter you license key.

Download and install Agisoft Metashape:
```bash
wget http://download.agisoft.com/metashape-pro_1_6_2_amd64.tar.gz
tar xvf metashape-pro_1_6_2_amd64.tar.gz
cd metashape-pro
LD_LIBRARY_PATH="python/lib:$LD_LIBRARY_PATH" ./python/bin/python3.5 -m pip install pillow
bash metashape.sh
```

Optionally, enable GPU acceleration by checking Tools -> Preferences -> GPU.

Depending on the specs of your PC you may need to downscale your images to proper size. We recommend using 4k images or less. For example, if you want to downscale images by a factor of two, run this command:
```bash
# convert comes with imagemagick package
# sudo apt install imagemagick

# in images directory
for fn in *jpg; do convert $fn -resize 50% $fn; done
```

Build point cloud:
```bash
bash metashape.sh -r <npbg>/scripts/metashape_build_cloud.py <my_scene>
```
where `<npbg>` is the path to NPBG sources, `<my_scene>` is directory with `images` subdirectory with your scene images.

The script will produce:
* `point_cloud.ply`: dense point cloud
* `cameras.xml`: camera registration data
* `images_undistorted`: undistorted images for descriptor fitting
* `project.psz`: Metashape project
* `scene.yaml`: scene configuration for the NPBG viewer

Make sure the point cloud has no severe misalignments and crop out unnecessary geometry to optimize memory consumption. To edit a scene, open `project.psz` in Metashape GUI and export modified point cloud (File -> Export -> Export Points). See **Issues** section for further recommendations.

Now we can fit descriptors for this scene.

#### Fitting descriptors

Modify `configs/paths_example.yaml` by setting absolute paths to scene configuration file, target images and, optionally, masks. Add other scenes to this file if needed.

Fit the scene:
```bash
python train.py --config configs/train_example.yaml --pipeline npbg.pipelines.ogl.TexturePipeline --dataset_names <scene_name>
```
where `<scene_name>` is the name of the scene in `paths_example.yaml`. Model checkpoints and Tensorboard logs will be stored in `data/logs`.

The command above will finetune weights of the rendering network. This regime usually produces more appealing results. To freeze the rendering network, use option `--freeze_net`. We provide pretrained weights for the rendering network on ScanNet and People dataset located in `downloads/weights`. Set pretrained network using `net_ckpt` option in `train_example.yaml`.

If you have masks for target images, use option '--use_masks'. Make sure masks align with target images.

When the model converge (usually 10 epochs is enough), run the scene in the viewer:

```bash
python viewer.py --config <my_scene>.yaml --checkpoint data/logs/<experiment>/checkpoints/<PointTexture>.pth --origin-view
```
where `<my_scene>.yaml` is the scene configuration file created in the point cloud reconstruction stage, `--checkpoint` is the path to descriptors checkpoint and `--origin-view` option automatically moves geometry origin to the world origin for convenient navigation. You can manually assign `model3d_origin` field in `<my_scene>.yaml` for arbitrary origin transformation (see `downloads/person_1.yaml` for example).

## Guidelines for fitting novel scenes

Fitting novel scenes can sometimes be tricky, most often due to the preparation of camera poses that are provided in different ways by different sources, or sometimes because of the reconstruction issues (see below). We recommend checking out [this](https://github.com/alievk/npbg/issues/2) and [this](https://github.com/alievk/npbg/issues/7) issues for detailed explanations.

The most important insight is related to the configs structure. There is a system of 3 configs used in NPBG:

<img src="docs/images/configs structure.png" width="600">

(there is another **optional** config -- *inference config*, which is essentially a *scene config* with `net_ckpt` and `texture_ckpt` parameters: paths to the network weights checkpoint and a descriptors checkpoint, respectively)

To fit a new scene, one should a *scene config* `configs/my_scene_name.yaml` and a *path config* `configs/my_scene_paths.yaml` by setting absolute paths to scene configuration file, target images, and other optional parameters, such as masks. *Path config* can contain paths to images of either 1 scene or several scenes, if needed. Examples of all configs of all types can be found in the repository.

## Code logic and structure

Since our repository is based on a custom, specific framework, we leave the following diagram with the basic code logic. For those who wish to extend our code with additional features or try out related ideas (which we would highly appreciate), this diagram should help finding where the changes should be applied in the code. At the same time, various technical intricacies are not shown here for the sake of clarity.

<img src="docs/images/code structure.png" width="900">


## Issues

* Reconstruction failure and geometry misalignment. Taking photos for photogrammetry is the most critical part in the whole pipeline. Follow these recommendations to have a good reconstruction:
  * Set the highest quality in the camera settings;
  * Keep the object in focus, don't set f-stop too low;
  * Fix shutter speed, the faster the better, but don't underexpose the photo;
  * Photos must be bright enough, but don't overexpose the photo;
  * Keep ISO low enough as it may introduce noise;
  * Keep the objects still, remove moving parts from the scene;
  * Take photos with at least 70% overlap;
  * If possible, use soft diffused light;
  * Avoid dark shadows;
  
  If you are using a smarthone with Android, OpenCamera may come handy. A good starting point for settings is f/8, ISO 300, shutter speed 1/125s. iPhone users are recommended to fix exposure in the Camera. Follow this [guide](https://github.com/alievk/npbg/blob/master/docs/Shooting%20best%20practices.md) for more recommendations.
* Viewer performance. If Pytorch and X server run on different GPUs there will be extra data transfer overhead between two GPUs. If higher framerate is desirable, make sure they run on the same GPU (use `CUDA_VISIBLE_DEVICES`).
* Pytorch crash on train. there is a known issue when Pytorch crashes on backward pass if there are different GPUs, f.e. GeForce GTX 1080 Ti and GeForce RTX 2080 Ti. Use `CUDA_VISIBLE_DEVICES` to mask GPU.

## TODOs

This is what we want to implement as well. We would also highly appreciate the help from the community.

* Point cloud reconstruction with COLMAP. As Agisoft Metashape is a proprietary software, the community would most benefit from an open source package like COLMAP which has almost the same functionality as Metashape, so the goal is to have `colmap_build_cloud.py` script working in the same manner as `metashape_build_cloud.py`.
* Convenient viewer navigation. Interactively choosing rotation center would make navigation much more conveniet. At the moment the viewer either explicitly imports the model's origin matrix or sets the origin automatically based on point cloud density.

## Citation

```
@article{Аliev2020,
    title={Neural Point-Based Graphics},
    author={Kara-Ali Aliev and Artem Sevastopolsky and Maria Kolos and Dmitry Ulyanov and Victor Lempitsky},
    year={2020},
    eprint={1906.08240v3},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
