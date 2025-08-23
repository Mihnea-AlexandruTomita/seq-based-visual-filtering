# Sequence-Based Filtering for Visual Route-Based Navigation

This repository provides sequence-based implementations of five Visual Place Recognition (VPR) techniques — <b>AMOSNet</b>, <b>HybridNet</b>, <b>CALC</b>, <b>HOG</b>, and <b>NetVLAD</b> — as described in the accompanying paper.

**Title:** Sequence-Based Filtering for Visual Route-Based Navigation: Analyzing the Benefits, Trade-Offs and Design Choices <br>
**Authors:** Mihnea-Alexandru Tomita, Mubariz Zaffar, Bruno Ferrarini, Michael J. Milford, Klaus D. McDonald-Maier and Shoaib Ehsan 

Published in IEEE Access, vol. 10, pp. 81974-81987, 2022 and available 📑 [here](https://doi.org/10.1109/ACCESS.2022.3196389).

The goal of this work is to systematically investigate the effects of sequence-based filtering on top of single-frame-based VPR techniques for route-based navigation. We analyze the trade-offs between accuracy and computational cost, examine the impact of sequence length, and identify combinations of techniques that deliver high performance efficiently. 

> **Note:** The sequence-based implementations of RegionVLAD and AlexNet are also provided in this repository for completeness, although they were not evaluated in the paper.

## 📂 Repository Structure
<pre>
├── AMOSNet/                   # AMOSNet supporting files – Add <b>AmosNet.caffemodel</b> in this folder
│   ├── ReadMe.txt             # Original AMOSNet citations
│   ├── amosnet_mean.npy       
│   └── deploy.prototxt 
├── AlexNet/                   # AlexNet files
│   ├── AlexNet_k.py           # Sequence-based AlexNet implementation
│   ├── ReadMe.txt             # Original AlexNet citations  
│   ├── __init__.py
│   └── alexnet/               # Network configuration folder - Add <b>alexnet.caffemodel</b> in this folder
│       └── deploy.prototxt  
├── CALC/                      # CALC supporting files
│   ├── ReadMe.txt             # Original CALC citations
│   ├── model/                 # Add calc.caffemodel in this folder
│   │   └── ReadMe.txt         
│   └── proto/                 # Contains network configuration
│       └── deploy.prototxt
├── HOG/                       # HOG files
│   ├── HOG_k.py               # Sequence-based HOG implementation
│   └── ReadMe.txt             # Original HOG citations
├── HybridNet/                 # HybridNet supporting files – Add <b>HybridNet.caffemodel</b> in this folder
│   ├── hybridnet_mean.npy     # Pretrained weights
│   ├── deploy.prototxt        # Network configuration
│   └── ReadMe.txt             # Original HybridNet citations
├── NetVLAD/                   # NetVLAD files
│   ├── __init__.py
│   ├── ReadMe.txt             # Original NetVLAD citations
│   ├── NetVLAD_k.py           # Sequence-based NetVLAD implementation
│   ├── checkpoints/           # Add the <b>checkpoints</b> for the NetVLAD model to this folder
│   │   └── ReadMe.txt         
│   └── netvlad_tf/            # Supporting TensorFlow modules
│       ├── .gitignore
│       ├── __init__.py
│       ├── image_descriptor.py
│       ├── layers.py
│       ├── mat_to_checkpoint.py
│       ├── net_from_mat.py
│       └── nets.py
├── RegionVLAD/                # RegionVLAD files
│   ├── RegionVLAD_k.py        # Sequence-based RegionVLAD implementation
│   ├── ReadMe.txt             # Original RegionVLAD citations
│   ├── __init__.py
│   ├── Vocabulary/            # Pre-computed vocabularies
│   │   ├── Vocabulary_100_200_300_Protocol2.pkl
│   │   ├── Vocabulary_100_200_300_Protocol3.pkl
│   │   ├── Vocabulary_400_Protocol2.pkl
│   │   └── Vocabulary_400_Protocol3.pkl
│   └── AlexnetPlaces365/      # Supporting files for AlexNet backbone – Add <b>alexnet_places365.caffemodel</b> to this folder
│       ├── deploy_alexnet_places365.prototxt
│       └── places365CNN_mean.binaryproto  
├── figures/ 
│   └── plot_figure3.py        # Script to reproduce Figure 3
├── AMOSNet_k.py               # Sequence-based AMOSNet implementation
├── CALC_k.py                  # Sequence-based CALC implementation
├── HybridNet_k.py             # Sequence-based HybridNet implementation 
├── MODEL_DOWNLOADS.md         # Markdown file containing download links and instructions for all pretrained models    
</pre>

## 📦 Pretrained Models

[MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md) provides the web locations and instructions for downloading all pretrained models required by the VPR techniques in this repository. These files are not included due to size restrictions.

## 🛠 Required Libraries By Technique:
- **AMOSNet:** `caffe`, `numpy`, `cv2`, `csv`       
- **CALC:** `caffe`, `numpy`, `cv2`, `csv`, `time`     
- **HOG:** `numpy`, `cv2`, `csv`
- **HybridNet:** `caffe`, `numpy`, `cv2`, `csv`                  
- **NetVLAD:** `tensorflow`, `numpy`, `cv2`, `csv`, `time`
- **RegionVLAD:** `caffe`, `numpy`, `cv2`, `csv`, `skimage`, `pickle`, `itertools`, `time`, `os`
- **AlexNet:** `caffe`, `numpy`, `cv2`, `csv`, `os`           


## 🚀 Running the Sequence-Based VPR Techniques
> **Note:** For each VPR technique in this repository, the Python files are named with a `_k` suffix (e.g. `NetVLAD_k.py`). This indicates that the implementation uses sequence-based filtering, where `k` represents the number of consecutive images in each sequence. The value of `k` can be adjusted for each technique. For more details, please refer to the accompanying paper.

Before running any `.py` file, update the following variables:
- `total_Query_Images` → Number of images in the **query folder**.
- `total_Ref_Images` → Number of images in the **reference folder**.
- `query_directory` → Path to your **query images** folder.
- `ref_directory` → Path to your **reference images** folder.
- `k` → Number of consecutive images in the sequence. Set `k = 1` to disable sequence-based filtering.

<pre>
total_Query_Images = 1000  # Number of images in the query folder
total_Ref_Images   = 1000  # Number of images in the reference folder

query_directory = '/path/to/query/images'
ref_directory   = '/path/to/reference/images'

k = 10  # Sequence length
</pre>

Once these variables are correctly set, you can run the corresponding `_k.py` file for your chosen technique.

## 📊 Datasets
The experiments use the following publicly available sequential VPR datasets:
- Campus Loop
- Gardens Point (day-to-day) 
- Gardens Point (day-to-night) 
- Nordland 

For more details on these datasets, including environmental conditions and experimental setup, please refer to our paper.

## 📌 Key Findings
- Sequence-based filtering consistently improves VPR accuracy by leveraging sequences of consecutive images.
- Lightweight descriptors such as HOG or CALC can sometimes outperform deep learning methods like NetVLAD while requiring less computation.
- Deep learning-based methods such as AMOSNet, HybridNet and NetVLAD are more robust under severe environmental and viewpoint changes but require more computational resources.
- Sequence-based matching may fail if the traversal speed between the query and reference sequences differ significantly.


## 📄 Citation

If you found this repository helpful, please cite the paper below:
<pre>
@ARTICLE{9849680,
  author={Tomiṭă, Mihnea-Alexandru and Zaffar, Mubariz and Ferrarini, Bruno and Milford, Michael J. and McDonald-Maier, Klaus D. and Ehsan, Shoaib},
  journal={IEEE Access}, 
  title={Sequence-Based Filtering for Visual Route-Based Navigation: Analyzing the Benefits, Trade-Offs and Design Choices}, 
  year={2022},
  volume={10},
  number={},
  pages={81974-81987},
  keywords={Filtering;Visualization;Navigation;Convolutional neural networks;Lighting;Image matching;Electronic mail;Sequence-based filtering;visual localization;visual place recognition},
  doi={10.1109/ACCESS.2022.3196389}}
 </pre>

## References for Evaluated VPR Techniques
The sequence-based filtering implementations in this repository build upon the single-frame-based VPR techniques that were evaluated in [VPR-Bench](https://github.com/MubarizZaffar/VPR-Bench) and described in the associated paper:

**Title:** VPR-Bench: An Open-Source Visual Place Recognition Evaluation Framework with Quantifiable Viewpoint and Appearance Change  
**Authors:** M Zaffar, S Garg, M Milford, J Kooij, D Flynn, K McDonald-Maier, S Ehsan  

Published in: International Journal of Computer Vision, 2021 and available [here](https://link.springer.com/article/10.1007/s11263-021-01469-5).

Citations for the original authors can be found in the `ReadMe.txt` file within each technique's folder (e.g. AMOSNet/ReadMe.txt, HOG/ReadMe.txt, etc.). All sequence-based filtering modifications in this repository were added on top of these original implementations, with proper credit given to the original authors for the backbone code.
