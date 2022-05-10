# A multi-task learning for cavitation detection and cavitation intensity recognition of valve acoustic signals 
[[arXiv]](https://arxiv.org/abs/2203.01118)

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
- Download trained models from [here](https://drive.google.com/drive/folders/14enrN8ZXC9a_7z_5mwkutsHVfV58dwUV?usp=sharing).
- Download datasets from [here](https://drive.google.com/drive/folders/1wCf2v1U1hNK_2sMGOitq-KyuQFk7rXOK?usp=sharing).



## Updates
[01.03.2022] We submit preprinted versions on the arXiv.

[08.03.2022] We upload all source codes for our method. And we add a link to our paper on the arXiv.

[19.04.2022] Our paper was accepted by the [[Engineering Applications of Artificial Intelligence (EAAI)]](https://www.journals.elsevier.com/engineering-applications-of-artificial-intelligence). The Impact Factor of this journal is 6.212, ranking it 7 out of 91 in Engineering, Multidisciplinary.

[10.05.2022] Our paper is now available online. And the link is [here](https://www.sciencedirect.com/science/article/pii/S0952197622001361?dgcid=coauthor).

For any queries, please feel free to contact YuSha et al. through yusha20211001@gmail.com

## Citation
If you find our work useful in your research, please consider citing:
```
@article{sha2022multi,
  title={A multi-task learning for cavitation detection and cavitation intensity recognition of valve acoustic signals},
  author={Sha, Yu and Faber, Johannes and Gou, Shuiping and Liu, Bo and Li, Wei and Shramm, Stefan and Stoecker, Horst and Steckenreiter, Thomas and Vnucec, Domagoj and Wetzstein, Nadine and Widl Andreas and Zhou Kai},
  journal={arXiv preprint arXiv:2203.01118},
  year={2022}
}
```
