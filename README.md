# A multi-task learning for cavitation detection and cavitation intensity recognition of valve acoustic signals (1D DHRN)
With the rapid development of smart manufacturing, data-driven machinery health management has received a growing attention. As one of the most popular methods in machinery health management, deep learning (DL) has achieved remarkable successes. However, due to the issues of limited samples and poor separability of different cavitation states of acoustic signals, which greatly hinder the eventual performance of DL modes for cavitation intensity recognition and cavitation detection. Also different task were performed separately conventionally. In this work, a novel multi-task learning framework for simultaneous cavitation detection and  cavitation intensity recognition framework using 1-D double hierarchical residual networks (1-D DHRN) is proposed for analyzing valves acoustic signals. Firstly, a data augmentation method based on sliding window with fast Fourier transform (Swin-FFT) is developed to alleviate the small-sample issue confronted in this study. Secondly, a 1-D double hierarchical residual block (1-D DHRB) is constructed to capture sensitive features from the frequency domain acoustic signals of valve. Then, a new structure of 1-D DHRN is proposed. Finally, the devised 1-D DHRN is evaluated on two datasets of valve acoustic signals without noise (Dataset 1 and Dataset 2) and one dataset of valve acoustic signals with realistic surrounding noise (Dataset 3) provided by SAMSON AG (Frankfurt). Our method has achieved state-of-the-art results. The prediction accurcies of 1-D DHRN for cavitation intensitys recognition are as high as 93.75%, 94.31% and 100%, which indicates that 1-D DHRN outperforms other DL models and conventional methods. At the same time, the testing accuracies of 1-D DHRN for cavitation detection are as high as 97.02%, 97.64% and 100%. In addition, 1-D DHRN has also been tested for different frequencies of samples and shows excellent results for frequency of samples that mobile phones can accommodate.
![img1](https://github.com/CavitationDetection/1-D-DHRN/blob/main/figs/1-D%20DHRN.png)
## Requirements

- Python 3.8.11
- torch 1.9.1
- torchvision 0.10.1

Note: our model is trained on NVIDIA GPU (A100).

## Code execution

- train.py is the entry point to the code.
- main.py is the main function of our model.
- network.py is the network structure of 1-D double hierarchical residual block (1-D DHRB) and 1-D double hierarchical residual network (1-D DHRN).
- opts.py is all the necessary parameters for our method (e.g. learning rate and data loading path and so on).
- Execute train.py

Note that, for the current version. test.py is nor required as the code calls the test function every iteration from within to visualize the performance difference between the baseline and the our method. 
- Download trained models from [here](https://drive.google.com/drive/folders/1ye8Vev8_fdMvdfHr5FIFSb5tcwtYHlnv) and place inside the directory ./models/
- Download datasets from [here](https://drive.google.com/drive/folders/1eejPrqM2hWPxSfb0gUhu-F4FD0rhO7sp?usp=sharing) and place test signals in the subdirectories of ./Data/Test/



## Updates
[1.3.2022] We submit preprinted versions on the arXiv.

[8.3.2022] We upload all source codes for our method.


For any queries, please feel free to contact YuSha et al. through yusha20211001@gmail.com

## Citation
If you find our work useful in your research, please consider citing:
```
@article{sha2022amultitask,
  title={A multi-task learning for cavitation detection and cavitation intensity recognition of valve acoustic signals},
  author={Sha, Yu and Faber, Johannes and Gou, Shuiping and Liu, Bo and Li, Wei and Shramm, Stefan and Stoecker, Horst and Steckenreiter, Thomas and Vnucec, Domagoj and Wetzstein, Nadine and Widl Andreas and Zhou Kai},
  journal={arXiv preprint arXiv:2203.01118},
  year={2022}
}
```
