<div align="center">
    <h1 align="center">NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction</h1>


[**Project Page**](https://open3dvlab.github.io/NeuRodin/) | [**Arxiv**](xx)

[Yifan Wang<sup>1</sup>](https://github.com/yyfz), [Di Huang<sup>1</sup>](https://dihuang.me/), [Weicai Ye<sup>1,2</sup>](https://ywcmaike.github.io/), [Guofeng Zhang<sup>2</sup>](http://www.cad.zju.edu.cn/home/gfzhang/) [Wanli Ouyang<sup>1</sup>](https://wlouyang.github.io/), [Tong He<sup>1</sup>](http://tonghe90.github.io/)

**<sup>1</sup>Shanghai AI Laboratory** &nbsp;&nbsp; **<sup>2</sup>State Key Lab of CAD&CG, Zhejiang University**
</div>

<p align="center">
    <img src="assets/Teaser.png" alt="overview" width="90%" />
</p>

This is the official implementation of paper "NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction".

Signed Distance Function (SDF)-based volume rendering has demonstrated significant capabilities in surface reconstruction. Although promising, SDF-based methods often fail to capture detailed geometric structures, resulting in visible defects. By comparing SDF-based volume rendering to density-based volume rendering, we identify two main factors within the SDF-based approach that degrade surface quality: SDF-to-density representation and geometric regularization. These factors introduce challenges that hinder the optimization of the SDF field. To address these issues, we introduce NeuRodin, a novel two-stage neural surface reconstruction framework that not only achieves high-fidelity surface reconstruction but also retains the flexible optimization characteristics of density-based methods. NeuRodin incorporates innovative strategies that facilitate transformation of arbitrary topologies and reduce artifacts associated with density bias. Extensive evaluations on the Tanks and Temples and ScanNet++ datasets demonstrate the superiority of NeuRodin, showing strong reconstruction capabilities for both indoor and outdoor environments using solely posed RGB captures. All codes and models will be made public upon acceptance.

## Results

<div align="center">






<figure class="half">
<img src="assets/barn.gif" alt="ballroom" style="display:inline-block; margin-right:10px;" width="47%">
    <img src="assets/ballroom.gif" alt="ballroom" style="display:inline-block; margin-right:10px;" width="47%">
</figure>

</div>

## Installation
Ensure that you have the following prerequisites installed on your system:
- conda
- CUDA 11.3


Create and Activate Conda Environment

First, create a new Conda environment named `neurodin` with Python 3.8:

```bash
conda create -n neurodin python==3.8 -y
```

Activate the newly created environment:
```bash
conda activate neurodin
```

Install PyTorch, TorchVision, and Torchaudio with CUDA 11.3 support (or with other CUDA versions and Torch versions. We tested 1.13.1+cu116, which also works):
```bash 
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install tiny-cuda-nn Bindings for PyTorch:
```bash 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install SDFStudio and command-line interface tab completion:
```bash
pip install -e .
ns-install-cli
```

To install PyMCubes for extracting meshes, use the following command:
```bash
pip install PyMCubes==0.1.4
```

## Data Preparation

### Tanks and Temples

For the training set, please download the camera poses and image set from [Tanks and Temples website](https://www.tanksandtemples.org/download/). Follow the [Neuralangelo data processing guide](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md) to preprocess the data.

The data should be arranged as:
```
- <data_root>
    - <scene_name>
        - images
        - transforms.json
        ...
```


For the advanced set, we directly use the data preprocessed by [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet?tab=readme-ov-file).

The data should be arranged as:
```
- <data_root>
    - <scene_name>
        - cams
        - images
```

### ScanNet++

Go to the [ScanNet++ website](https://kaldir.vc.in.tum.de/scannetpp/) and download the DSLR dataset. Use the [script from sdfstudio](https://github.com/autonomousvision/sdfstudio/blob/master/scripts/datasets/process_nerfstudio_to_sdfstudio.py) to convert it to the sdfstudio format.

The data should be arranged as:
```
- <data_root>
    - <scene_name>
        - *.png
        - meta_data.json
```

### Custom Data
For custom data, we suggest organizing it in the COLMAP format, following the instructions provided in the [Neuralangelo data processing guide](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md). Alternatively, you can write a custom dataparser script for NeRFStudio/SDFStudio based on your own data structure.

## Running
We have configs for room-level indoor scene (ScanNet++), large scale indoor and outdoor scene (Tanks and Temples).

### Example for Tanks and Temples
#### Training Set
Outdoor scene:
```bash
# Stage 1
ns-train neurodin-stage1-outdoor-large --experiment_name neurodin-Barn-stage1 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 tnt-data --data <path-to-tnt> --scene_name Barn
 
# Stage 2
ns-train neurodin-stage2-outdoor-large --experiment_name neurodin-Barn-stage2 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 --trainer.load_dir <path-to-stage1-checkpoints-dir> tnt-data --data <path-to-tnt> --scene_name Barn
```

Indoor scene:
```bash
# Stage 1
ns-train neurodin-stage1-indoor-large --experiment_name neurodin-Meetingroom-stage1 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 tnt-data --data <path-to-tnt> --scene_name Meetingroom
 
# Stage 2
ns-train neurodin-stage2-indoor-large --experiment_name neurodin-Meetingroom-stage2 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 --trainer.load_dir <path-to-stage1-checkpoints-dir> tnt-data --data <path-to-tnt> --scene_name Meetingroom
```

#### Advance Set
Indoor scene:
```bash
# Stage 1
ns-train neurodin-stage1-indoor-large --experiment_name neurodin-Ballroom-stage1 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 tnt-advance-data --data <path-to-tnt> --scene_name Ballroom
 
# Stage 2
ns-train neurodin-stage2-indoor-large --experiment_name neurodin-Ballroom-stage2 --pipeline.datamanager.eval_camera_res_scale_factor 0.5 --trainer.load_dir <path-to-stage1-checkpoints-dir> tnt-advance-data --data <path-to-tnt> --scene_name Ballroom
```

### Example for ScanNet++
```bash
# Stage 1
ns-train neurodin-stage1-indoor-small --experiment_name neurodin-21d970d8de-stage1 --pipeline.datamanager.camera_res_scale_factor 0.5 sdfstudio-data --data data/21d970d8de --scale_factor 0.8 

# Stage 2
ns-train neurodin-stage2-indoor-small --experiment_name neurodin-21d970d8de-stage2 --trainer.load_dir <path-to-stage1-checkpoints-dir> --pipeline.datamanager.camera_res_scale_factor 0.5 sdfstudio-data --data data/21d970d8de --scale_factor 0.8
```

## Evaluation
We recommend using `zoo/extract_surface.py` (adapted from [Neuralangelo](https://github.com/NVlabs/neuralangelo)) to extract the mesh. This method is faster because it doesn't require loading all images as `ns-extract-mesh` in sdfstudio does.

```bash
python zoo/extract_surface.py --conf <path-to-config> --resolution 2048
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{wang2024neurodin,
  title={NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction.},
  author={Yifan Wang and Di Huang and Weicai Ye and Guofeng Zhang and Wanli Ouyang and Tong He},
    booktitle={arxiv preprint}, 
    year={2024}
}
```

## Acknowledgement
This codebase is modified from [SDFStudio](https://github.com/autonomousvision/sdfstudio), [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/) and [Neuralangelo](https://github.com/NVlabs/neuralangelo). Thanks to all of these great projects.
