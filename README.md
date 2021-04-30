
# LENNA
03.2020 - 06. 2020  | zenith-lee, pyossyoung, meowpunch | *supported by Accelerated Computing Systems Lab, Yonsei Univ.*

LENNA (Latency Estimation for Neural Network Architecture) upgrades Differentiable Architecture Search (DARTS), which is known as high performance model in Neural Architecture Search (NAS)

## PROGRESS
All progress is in [Notion Pages [KR]](https://www.notion.so/f44624493796475984f9651728b225d7)

## ABSTRACT
### Introduce
These days, researches on NAS (Representative methodology of AutoML) that has hit the Artificial Intelligence (AI) field are being actively carried out. 
However, most researches are far from being practical and are focused only on performance metrics such as accuracy. 
So, we can search a practical architecture that can be used in real life by adding hardware metrics such as latency to the loss function.

### Related Works
ProxylessNAS searches architecture considering the target hardware metrics. 
But, ProxylessNAS is applied to simplified structure with parallel arranged operations and has limits to be applicable to general complex architecture structure such as DARTS-made structure.
We introduce LENNA, the Multi-Layer Perceptron model made for estimating latency given fundamental information of network, such as parameters, input size, etc. 

The latency part of newly generated DARTS loss function would be estimated by LENNA.
- 𝑳𝒐𝒔𝒔 = 𝑳𝒐𝒔𝒔(𝑫𝑨𝑹𝑻𝑺)+ 𝝀 ∗ (𝒆𝒙𝒑𝒆𝒄𝒕𝒆𝒅 𝒍𝒂𝒕𝒆𝒏𝒄𝒚)
<img width="100%" alt="image" src="https://user-images.githubusercontent.com/40639955/116496223-85bf1d00-a8df-11eb-8a45-19519e006c8d.png">
  
## EXPERIEMNT
### Env
- CPU: AMD Ryzen 7 3700X 8-core Processor * 16 
- GPU: GeForce RTX 2060 SUPER * 4

### Structure
the project includes followings:
- submission
  - pre-practice by using ElasticNet
  - `L(one block) = sum(L(op))`
- generate dataset
- preprocessing
- modeling

### Methodology
- How many times do we need to estimate latency when resetting the binary gate?
- How to handle poping values .
- L(one block) = sum(latency of ops of the block)




## REFERENCES
1. Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. ICLR, 2019.
2. Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar. Designing neural network architectures using reinforcement learning. ICLR, 2017.
3. Jelena Luketina, Mathias Berglund, Klaus Greff, and Tapani Raiko. Scalable gradient-based tuning of continuous regularization hyperparameters. In ICML, pp. 2952–2960, 2016.
4. Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. ICLR, 2017
5. Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. arXiv preprint arXiv:1707.01083, 2017. 
6. Han Cai, Ligeng Zhu, Song Han, ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. ICLR, 2019.
7. Liu, C., Zoph, B., Shlens, J., Hua, W., Li, L.-J., FeiFei, L., Yuille, A., Huang, J., and Murphy, K. Progressive neural architecture search. ECCV, 2018.
8. Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning transferable architectures for scalable image recognition. In CVPR, 2018.
