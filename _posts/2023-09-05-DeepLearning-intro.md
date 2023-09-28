---
layout: single

title: "딥러닝 소개"

date: 2023-09-05 20:00:00 +0900
lastmod: 2023-09-05 20:00:00 +0900 # sitemap.xml에서 사용됨

author_profile: true

header:
  overlay_image: https://user-images.githubusercontent.com/91832324/229981423-3a9c5404-403f-4ea8-a721-e941b2f41866.jpg

  overlay_filter: 0.5 # 투명도

categories: 
  - College Deep Learning 

tags: 
    - Deep Learning 
    - Python
    - College

# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "table of content" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---
# 1. 딥러닝 소개

# 딥러닝 소개

## 머신러닝/딥러닝

- 머신러닝/딥러닝은 데이터 분석 및 인공지능 관련 기술을 획기적으로 발전시켰음.
- 사회, 경제, 산업의 거의 모든 분야에서 머신러닝/딥러닝의 활용이 점차 강조되어 왔음.
- 대형 언어 모델(LLM, Large language model)인 GPT(Generative pre-trained transformer)의 출현으로
    
    딥러닝에 대한 관심이 보다 깊어졌음.
    
- 머신러닝/딥러닝 기술이 이미 많이 대중화되었음.

### 강의 주제

- 딥러닝의 기본 개념과 함께 다양한 딥러닝 기법을 최대한 직관적으로 전달하는 일에 집중
- 텐서플로우$$_{TensorFlow} 2$$ 와 케라스$$_{Keras}$$ 활용

## 주요 내용

- 딥러닝 개념 소개
- 딥러닝 활용
    - 컴퓨터비전: 이미지 분석 및 분할
    - 시계열 예측
    - 자연어 처리: 텍스트 분류
    - 생성 신경망 모델: 문장, 이미지 등등 생성

## 인공지능, 머신러닝, 딥러닝

### 관계 1: 연구 분야 관점

![](/assets/image/DeepLearning/01/Untitled.png)

그림 출처: [교보문구](https://www.kyobobook.co.kr/readIT/readITColumnView.laf?thmId=00198&sntnId=14142)

인공지능이 머신러닝을 반드시 사용하는건 아니다.

머신러닝을깊게 파고들면 딥러닝이 있다.

데이터 과학은 지표를 비롯한 통계를 뜻한다.

인터넷이 발전함에 따라 데이터가 방대해져서 통계를 계산하기가 까다로워져서 머신러닝과 딥러닝을 사용해 통계를 계산한다.

### 관계 2: 역사

![](/assets/image/DeepLearning/01/Untitled 1.png)

그림 출처: [NVIDIA](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/)

### 인공지능

- 인공지능: 인간의 지적 활동을 모방하여 컴퓨터로 자동화하려는 시도. 머신러닝과 딥러닝을 포괄함.
- (1950년대) 컴퓨터가 생각할 수 있는가? 라는 질문에서 출발
- (1980년대까지) **학습**(러닝)이 아닌 모든 가능성을 논리적으로 전개하는 기법 활용
    - 서양장기(체스) 등에서 우수한 성능 발휘
    - 반면에 이미지 분류, 음석 인식, 자연어 번역 등 보다 복잡한 문제는 제대로 다루지 못함.
- (1990 년대부터) 입력 데이터로부터 규칙을 스스로 찾아내도록 유도하는 머신러닝 기법이 유행하기 시작함.
    - 인공지능(AI) 분야의 주요 핵심 기법으로 자리잡음

### 전통적 프로그래밍 vs 머신러닝

![](/assets/image/DeepLearning/01/Untitled 2.png)

그림 출처: [MANNING](https://www.manning.com/books/deep-learning-with-python-second-edition)

### 머신러닝 모델 학습의 필수 요소

- 입력 데이터셋(훈련셋)
- 타깃 데이터셋
- 모델 평가지표

### 머신러닝 vs 딥러닝

- 머신러닝은 사전에 정의된 특징을 기반으로 모델을 학습한다.
- 딥러닝은 데이터로부터 표현을 스스로 학습하여 모델이 높은 수준의 추상적인 특징을 자동으로 추출하는 데 중점을 둔다.
- 딥러닝의 표현법은 아래 잘 설명되어 있음

### 학습 규칙과 데이터 표현법

- 데이터 표현법 학습: 주어진 과제 해결에 가장 적절한 **데이터 표현법**을 모델 학습을 통해 알아냄.
- 예제: 좌표 변환

![](/assets/image/DeepLearning/01/Untitled 3.png)

그림 출처: [MANNING](https://www.manning.com/books/deep-learning-with-python-second-edition)

1. 입력된 데이터는 $$(x_1, x_2)$$였으나 최종적인 3. 데이터 표현법에서는 $$(y_1,y_2)$$로 표현이 변환된것을 확인할 수 있다.

### 데이터 변환 자동화

- 반면에 MNIST 손글씨의 경우는 좋은 표현법을 수동으로 찾는 일은 거의 불가능
    
    ![](/assets/image/DeepLearning/01/Untitled 4.png)
    
    그림 출처: [한빛출판네트워크](https://www.hanbit.co.kr/store/books/look.php?p_code=B9267655530)
    
- 머신러닝 모델은 아래에 언급된 변환 등을 조합하여 문제 해결에 보다 도움이 되는 표현 변환 알고리즘을 찾아내려 시도함.
    - 회전
    - 이동
    - 사영$$_{projection}$$
    - 잘라내기

### 가설 공간

- 주어진 문제의 해결에 가장 적절한 변환 알고리즘을 머신러닝 모델 스스로 알아내기는 기본적으로 불가능
- 대신에 데이터 표현법 변환 알고리즘을 어떻게 구현할 수 있는지 길안내 필요. 즉, 머신러닝 모델의 구성은 사람이 직접 지정
- 그러면 가능한 모든 알고리즘의 공간 내에서 최적의 데이터 변환 알고리즘을 데이터 학습을 통해 알아냄.
- 가설 공간: **머신러닝 모델이 변환 알고리즘 학습에 활용할 수 있는 알고리즘의 공간**

### 딥러닝 모델

- 딥러닝 모델은 세 개 이상의 층으로 구성된 **심층 신경망**으로 구현
- ImageNet 분류 심층 신경망 모델 (2012년)

![](/assets/image/DeepLearning/01/Untitled 5.png)

- 층이 너무 많아도 과적합 문제가 발생할 수 있음
- 어떤 문제를 주고 해당 문제의 답을 구하는 신경망을 컴퓨터가 직접 구하는건 불가능함
(튜링 정지 문제의 핵심)
- 딥러닝을 통해 어떤 문제든 해결할 수 있는가?
    - 그런건 불가능
    - why? - 튜링의 정지 문제를 참고

그림 출처: [NerulPS Proceddings](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

### 예제: 손글씨 숫자 인식

- 네 개의 층 사용
- 층마다 입력 데이터 변환
- 최종적으로 **(사람은 이해할 수 없지만)** 모델은 어떻게든 적절한 예측값을 계산할 수 있는 표현법으로 데이터 변환
    - Layer 3을 보면 사람이 봤을때는 알 수 없으나 딥러닝은 이를 특징으로 인식하고 분류가 가능함

![](/assets/image/DeepLearning/01/Untitled 6.png)

그림 출처: [MANNING](https://www.manning.com/books/deep-learning-with-python-second-edition)

## 머신러닝 역사

### 초창기 신경망

- 신경망의 기본 아이디어: 1950년대부터
- 최초의 성공적인 신경망 활용: 1989년 미국의 벨 연구소의 얀 르쿤$$_{Yann LeCun}$$이 **LeNet 합성곱 신경망**
    
    ![](/assets/image/DeepLearning/01/homl14-16.gif)
    

그림 출처: [https://yann.lecun.com/exdb/lenet/index.html](https://yann.lecun.com/exdb/lenet/index.html)

### 결정트리, 랜덤 포레스트, 그레이디언트 부스팅

- 2000년대: **결정트리$_{decision tree}$**
- 2010년: **랜덤 포레스트$_{random forest}$**
- 2014년: **그레이디언트 부스팅$_{gradient boosting}$** 기법
- 현재까지도 딥러닝과 더불어 가장 많이 활용됨

### 딥러닝의 본격적 발전

- 2011년: GPU를 활용한 딥러닝 모델 훈련이 시작
- 2012년: 이미지 분류 경진대회인 [이미지넷의 ILSVRC](https://www.image-net.org/challenges/LSVRC/index.php)의
    
    2012년 대회에서 소개된 합성곱 신경망(CNN) 모델의 성능 매우 뛰어남
    
    - NVIDIA에서 GPU를 개발하기 시작해서 획기적으로 발전함

![](/assets/image/DeepLearning/01/Untitled 7.png)

그림 출처: [bulentsiyah](https://www.bulentsiyah.com/imagenet-winning-cnn-architectures-ilsvrc)

### 최근 머신러닝 분야 동향

![](/assets/image/DeepLearning/01/Untitled 8.png)

### 딥러닝 발전 동력

- 하드웨어
    - 2011년도 - GPU 개발 시작
- 데이터
    - 인터넷의 발달로 인한 데이터가 과거와 다르게 풍부해짐
- 알고리즘
    - 신경망 구조의 발전