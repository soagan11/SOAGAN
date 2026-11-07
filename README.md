# SOAGAN

## The dual-recognition generative adversarial network for image super-resolution

## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.5.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.


## How to Train

1. Download DIV2K and Flickr2K according the paper data avaliable section.
```
python train.py -opt options/train/train_SOAGAN_PSNR_x4.yml
```

## How to Test
1. Download All datasets (Set5, Set14, B100, Urban100, Manga109) according the paper data avaliable section
2. Run test. 
```
python test.py -opt option/test/SOAFAN/test_SOAGAN_x4.yml
```


## Contact   
For any inquiries or collaboration opportunities, please contact the author at lhl1231009@126.com.

## Acknowledgements   
We would like to thank all the contributors and supporters of this project and the avaliable BasicSR project.







