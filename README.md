
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

### Structure
the project includes followings:
- submission
  - pre-practice by using ElasticNet
  - `L(one block) = sum(L(op))`
- generate dataset
- preprocessing
- modeling

  
## EXPERIMENT

### Environment
- CPU: AMD Ryzen 7 3700X 8-core Processor * 16 
- GPU: GeForce RTX 2060 SUPER * 4

### Data Generator

batch_size: 64

#### Input X
num_layer 5 기준 167 dimension

- block type (need to be one hot encoded): normal(1), reduction(0)
  - 16 32 32 → 16 32 32 normal
  - 16 32 32 → 32 16 16 reduction

- input_channel: 1~1000 (Caution `RuntimeError: CUDA out of memory`)

- arhitecture parameters: random on unifrom distribution
  

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4bc9216-307e-47de-ae9b-0d9c3d629f9d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4bc9216-307e-47de-ae9b-0d9c3d629f9d/Untitled.png)

[Init ratio for arch param [KR]](https://www.notion.so/Init-ratio-for-arch-param-d331d8f8c9434c268be8434c406ed8b8)

[How to generate param? [KR]](https://www.notion.so/How-to-generate-param-218f7c8bb3c04c2891f280be837456a9)

#### Target y

##### ALGORITHM (about one row)

analysis.binary_gates 중

1. binary gate를 reset할 때 마다, latency를 10번 재서 40% 값을 사용해 모은다. (그림 1) 
  - [How many times do you need to estimate latency when resetting the binary gate? [KR]](https://www.notion.so/How-many-times-do-you-need-to-estimate-latency-when-resetting-the-binary-gate-9c138302f8fc408b8eac9129d0658a11)
2. 누적된 latency의 평균을 구한다. (그림 2)
3. 누적 평균의 전 후 오차를 계산한다. (그림 3)
4. 오차값이 누적 평균의 1% 이하로 연속해서 10번 hit하면 누적 평균 값을 사용한다.

##### SNAPSHOT

latency_by_binary_gates.ipynb 중 

- cumulative latency

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fbe1f282-5e2c-456e-b67d-34ea91ee42c0/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fbe1f282-5e2c-456e-b67d-34ea91ee42c0/Untitled.png)

그림 1>  누적 시간

- cumulative average

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1aadc90-19c2-43db-b4dc-b1665b13bf9f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1aadc90-19c2-43db-b4dc-b1665b13bf9f/Untitled.png)

그림 2> 누적 평균

- cumulative error

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ab539546-d2db-48d5-a5bb-50cc2ff4c02d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ab539546-d2db-48d5-a5bb-50cc2ff4c02d/Untitled.png)

그림 3>  누적 평균 오차

'그림 1'에서 binary gate를 reset할 때마다 측정한 latency는 항상 일정하지 않는 값을 가진다. 이렇게 측정한 latency를 누적하여 평균을 계속 구하면 일정 값으로 수렴하는 것을 확인 할 수 있고(그림 2)  이 누적 평균의 가장 최근 값과 그 직전 값의 오차는 점차 줄어드는 것을 확인 할 수 있다.(그림 3)


### Methodology
- How many times do we need to estimate latency when resetting the binary gate?
- How to handle poping values .
- L(one block) = sum(latency of ops of the block)






### Modeling
- Elastic Net

  <img width="50%" alt="image" src="https://user-images.githubusercontent.com/40639955/116647275-19165200-a9b5-11eb-880b-853d6e60f878.jpg">

- MLP Regressor

  <img width="50%" alt="image" src="https://user-images.githubusercontent.com/40639955/116647291-216e8d00-a9b5-11eb-873f-9e608b9fe67e.jpg">




## REFERENCES
1. Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. ICLR, 2019.
2. Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar. Designing neural network architectures using reinforcement learning. ICLR, 2017.
3. Jelena Luketina, Mathias Berglund, Klaus Greff, and Tapani Raiko. Scalable gradient-based tuning of continuous regularization hyperparameters. In ICML, pp. 2952–2960, 2016.
4. Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. ICLR, 2017
5. Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. arXiv preprint arXiv:1707.01083, 2017. 
6. Han Cai, Ligeng Zhu, Song Han, ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware. ICLR, 2019.
7. Liu, C., Zoph, B., Shlens, J., Hua, W., Li, L.-J., FeiFei, L., Yuille, A., Huang, J., and Murphy, K. Progressive neural architecture search. ECCV, 2018.
8. Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning transferable architectures for scalable image recognition. In CVPR, 2018.
