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
