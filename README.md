
# LENNA
03.2020 - 06. 2020  | zenith-lee, pyossyoung, meowpunch | *supported by Accelerated Computing Systems Lab, Yonsei Univ.*

LENNA (Latency Estimation for Neural Network Architecture) upgrades Differentiable Architecture Search (DARTS), which is known as high performance model in Neural Architecture Search (NAS)

## PROGRESS
All progress is in [Notion Pages KR](https://www.notion.so/f44624493796475984f9651728b225d7)

## ABSTRACTION

### Introduction
The project upgrades Differentiable Architecture Search (DARTS), which is known as high performance model in Neural Architecture Search (NAS), by a new, never before method. 
DARTS addresses comparable technique with state-of-the-art NAS (Neural Architecture Search) but has a latent flaw that it does not take direct metrics, such as latency, into accounting for model design.
Our method tries to improve hardware performance by attaching latency element into loss function of DARTS, thereby makes DARTS of more efficiency that also takes latency into account using gradient search. 
It seeks to improve performance by upgrading DARTS, which used to be well-known NAS model, and in the process, it does not simply measure hardware metrics but uses expected latency value by using self-made deep learning models.

### Overview
<img width="100%" alt="image" src="https://user-images.githubusercontent.com/40639955/116496223-85bf1d00-a8df-11eb-8a45-19519e006c8d.png">
`𝑳𝒐𝒔𝒔 = 𝑳𝒐𝒔𝒔(𝑫𝑨𝑹𝑻𝑺)+ 𝝀 ∗ (𝒆𝒙𝒑𝒆𝒄𝒕𝒆𝒅 𝒍𝒂𝒕𝒆𝒏𝒄𝒚)`



## EXPERIEMNT

###  STRUCTURE
the project includes followings:
- submission
  - `L(one block) = sum(L(op))`
- generate dataset
- preprocessing
- modeling
- 
### Env
CPU: AMD Ryzen 7 3700X 8-core Processor * 16 
GPU: GeForce RTX 2060 SUPER * 4
<img width="412" alt="image" src="https://user-images.githubusercontent.com/40639955/116496536-3f1df280-a8e0-11eb-803c-7181e37af395.png">


## REFERENCES
1. Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. ICLR, 2019.
2. Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar. Designing neural network architectures using reinforcement learning. ICLR, 2017.
3. Jelena Luketina, Mathias Berglund, Klaus Greff, and Tapani Raiko. Scalable gradient-based tuning of continuous regularization hyperparameters. In ICML, pp. 2952–2960, 2016.
4. Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. ICLR, 2017
5. Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. arXiv preprint arXiv:1707.01083, 2017. 
6. Han Cai, Ligeng Zhu, Song Han, ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. ICLR, 2019.
7. Liu, C., Zoph, B., Shlens, J., Hua, W., Li, L.-J., FeiFei, L., Yuille, A., Huang, J., and Murphy, K. Progressive neural architecture search. ECCV, 2018.
8. Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning transferable architectures for scalable image recognition. In CVPR, 2018.
