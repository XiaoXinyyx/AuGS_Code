# Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting
Yixin Yang, Bojian Wu, Yang Zhou, Hui Huang 
<br>
| [Project Page](https://xiaoxinyyx.github.io/AuGS/) | [Full Paper](https://xiaoxinyyx.github.io/AuGS/static/images/AuGS_ICLR2026.pdf) | 
<br>

![Teaser image](assets/teaser.png)

## Installation

Create virtual environment.
```shell
conda create -n AuGS python=3.9
conda activate AuGS
```

Install pytorch.
```shell
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install dependencies.
```shell
pip install -r requirements.txt
```

Install submodules
```shell
pip install ./diff-gaussian-2d-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
pip install ./submodules/fused-ssim --no-build-isolation
pip install ./submodules/gsplat --no-build-isolation
pip install ./submodules/wpca
```

## Results reproduction

To reproduce our results in the paper, run
```shell
python benchmark.py --mcmc \
    -m360 [path to mipnerf360 dataset] \
    -ns [path to nerfsynthetic dataset] \
    -tnt [path to tanks and temples dataset] \
    -db [path to deepblending dataset]
```

**Parameters:**
- `--mcmc`: When added, the base scene is trained with 3DGS-MCMC. Without this flag, 3DGS is used for base scene training.
- `--sh_degree`: Spherical harmonics degree for scene representation.

## Training Your Own Scenes

### 1. Training 3D Gaussian Splatting (Stage 1)

Train the initial 3D Gaussian model:

```shell
python train_3dgs.py \
    -s /path/to/your_dataset \
    -m output/your_scene \
    --eval \
    --sh_degree 3 \
    --iterations 30000 \
    --disable_viewer
```



For the MCMC variant (used in our paper):
```shell
python train_mcmc.py \
    -s /path/to/your_dataset \
    -m output/your_scene \
    --eval \
    --sh_degree 3 \
    --iterations 30000 \
    --cap_max [estimated_gaussian_count] \
    --opacity_lr 0.05 \ 
    --densify_until_iter 25000 \
    --disable_viewer
```
**Key parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-s` | Path to source dataset | Required |
| `-m` | Path to output model | Required |
| `--sh_degree` | Spherical harmonics degree | 3 |
| `--iterations` | Number of training iterations | 30000 |
| `--white_background` | Use white background | False |
| `--resolution` | Image resolution multiplier | -1 |
| `--cap_max` | Maximum number of Gaussians (for MCMC mode) | -1 |

### 2. Training 2D Kernels and Projection (Stage 2)

After Stage 1 completes, train and project 2D refinement kernels to 3D space:

```shell
python train_2d.py \
    -m output/your_scene \
    --ratio 0.1
```

**Key parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ratio` | Ratio of 2D kernels to original Gaussians | 0.1 |
| `--workers` | Number of CPU workers for projection | 16 |
| `--skip_train_2d` | Skip 2D training, only do projection |  |

### 3. Refining 3D Gaussians (Stage 3)

Refine the augmented Gaussian model from Stage 2:

```shell
python refine.py \
    --source_path /data/my_scene \
    -m output/my_scene \
    --start_checkpoint output/my_scene/point_cloud/iteration_30000/aug_point_cloud_init.ply \
    --eval \
    --iterations [30000 + num_cameras * 30] \
    --sh_degree 3 \
    --position_lr_init 0.000016 \
    --position_lr_final 0.000016 \
    --feature_lr 0.001 \
    --opacity_lr 0.02 \
    --scaling_lr 0.002 \
    --rotation_lr 0.0005 \
    --opadir_lr 0.001 \
    --theta_lr 0.0002 \
    --beta_lr 0.002
```

### 4. Rendering and Evaluation

Render novel views and compute metrics:

```shell
python render.py \
    --source_path /path/to/your_dataset \
    -m output/your_scene \
    --eval \
    --skip_train \
    --iterations [30000 + num_cameras * 30]
```


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{
    yang2026augmented,
    title={Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting},
    author={Yixin Yang and Bojian Wu and Yang Zhou and Hui Huang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://xiaoxinyyx.github.io/AuGS/}
}</code></pre>
  </div>
</section>



## Funding and Acknowledgments

This work was supported in parts by the National Key R&D Program of China (2024YFB3908500,2024YFB3908502), NSFC(U21B2023), Guangdong Basic and Applied Basic Research Foundation (2023B1515120026), and Scientific Development Funds from Shenzhen University.
