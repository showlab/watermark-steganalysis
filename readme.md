<div align="center">
<h1>🔓 Can Simple Averaging Defeat Modern Watermarks? 🤔</h1>
</div>

<div align="center">
    Pei Yang<sup>&#42;</sup>&nbsp;, <a href="https://scholar.google.com/citations?user=GMrjppAAAAAJ&hl=en">Hai Ci</a><sup>&#42;</sup>&nbsp;, <a href="https://scholar.google.com/citations?user=L2YS0jgAAAAJ&hl=en&oi=ao">Yiren Song</a>&nbsp;, and <a href="https://sites.google.com/view/showlab">Mike Zheng Shou</a><sup>&#x2709</sup>

</div>

<div align="center">
    <a href='https://sites.google.com/view/showlab/home?authuser=0' target='_blank'>Show Lab, National University of Singapore</a>
    <p>
</div>

<div align="center">
    <a href="https://arxiv.org/abs/2406.09026">
        <img src="https://img.shields.io/badge/arXiv-2406.09026-b31b1b.svg" alt="arXiv">
    </a>
    <p>
</div>

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2406.09026-b31b1b.svg)](https://arxiv.org/abs/2406.09026) -->


<img src="assets/maginot_line.png" width="1024">


## 🧰 Security Guidelines

We call for future watermarking research to benchmark their methods against simple steganalysis using our provided code. Getting started:

### Dependencies

```bash
pip install numpy<2 Pillow matplotlib tqdm
```

### Benchmark Code Usage

```bash
python benchmark.py \
    --watermark_method RingID \
    --width 512 \
    --height 512 \
    --ood_clean_path ~/Datasets/ImageNet/test \
    --ind_clean_path ~/Datasets/RingID/clean \
    --watermarked_path ~/Datasets/RingID/watermarked \
    --output_path /path/to/save/images \
    --num_eval_images 100
```

`ind_clean_path` and `watermarked_path` are paired paths to non-watermarked and watermarked images. If `123456.png` is in `ind_clean_path`, then its watermarked pixel-aligned counterpart should present in `watermarked_path` with exactly the same filename `123456.png`.
```
ood_clean_path/              # 5000+ original images from another dataset
├── 000000.png
├── 000001.png
├── 000002.png
└── ...

ind_clean_path/              # 5000+ paired non-watermarked images
├── ringid_0000.png
├── ringid_9801.png          # File names should precisely match, images should be pixel-aligned
└── ...

watermarked_path/            # 5000+ paired watermarked images
├── ringid_0000.png
├── ringid_9801.png          # File names should precisely match, images should be pixel-aligned
└── ...
```


## 🔑 Patterns Averaged Out

<img src="assets/content_agnostic_patterns.png" width="1024">

## 🚀 Open-Source
- [x] Core benchmark code for watermark removal/forgery 🧰
- [ ] Images we used during experiments
- [ ] Complete experiment code (currently being organised; unpolished version available on request: contact yangpei@comp.nus.edu.sg for access)

## Citation

```
@misc{yang2024steganalysisdigitalwatermarkingdefense,
      title={Steganalysis on Digital Watermarking: Is Your Defense Truly Impervious?}, 
      author={Pei Yang and Hai Ci and Yiren Song and Mike Zheng Shou},
      year={2024},
      eprint={2406.09026},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.09026}, 
}
```





