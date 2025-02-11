# Train SAM 2: Train/Fine-Tune Segment Anything 2 (Guide/code)
This repository contains tutorial code for fine-tuning/training segment anything 2.

You can find the [full toturial associate with code at this LINK](https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3).
 

The training script can be found in [train.py](https://github.com/bekhzod-olimov/Fine-Tune-Segment-Anything-Model-SAM2/blob/main/train.py) and should work as is with the [LabPics 1 dataset](https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1).

The code for loading and using the fine-tuned net can be found in [inference.py](https://github.com/bekhzod-olimov/Fine-Tune-Segment-Anything-Model-SAM2/blob/main/inference.py)

Other than these two files no modification was done to the SAM2 repository.

The training is done on the [LabPics 1 dataset](https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1). 


## Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
