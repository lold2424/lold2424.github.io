---
layout: single

title: "[Computer Network] 5장 아날로그 전송"

date: 2023-04-11 13:00:00 +0900
lastmod: 2023-04-11 13:00:00 +0900 # sitemap.xml에서 사용됨

author_profile: true

header:
  overlay_image: https://www.google.com/url?sa=i&url=https%3A%2F%2Fpixabay.com%2Fko%2Fillustrations%2F%25EB%258D%25B0%25EC%259D%25B4%25ED%2584%25B0-%25ED%2594%2584%25EB%25A1%259C%25EA%25B7%25B8%25EB%259E%25A8-%25EC%259E%2591%25EC%2584%25B1-7542750%2F&psig=AOvVaw24x7S74ZGyWwKglY2UfCBs&ust=1682404459558000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCNCippPzwf4CFQAAAAAdAAAAABAE

  overlay_filter: 0.5 # 투명도

categories: 
  - College Computer Network

tags: 
    - Computer Network
    - College

# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "주요 목차" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---
# 5. 아날로그 전송

## 1. 디지털-대-아날로그 변환

디지털-대-아날로그 변환은 디지털 데이터의 정보를 기반으로 아날로그 신호의 특성 중 하나를 변경하는 처리이다.

![Untitled](https://user-images.githubusercontent.com/91832324/233918031-94a36ab4-0b3b-4864-a752-b2d1a36a08e4.png)


- **반송파 신호(Carrier Signal)**
    - 통신에서 정보의 전달을 위해 입력 신호를 변조한 전자기파(일반적으로 사인파)를 의미한다.
        - 변조 : 신호의 진폭, 주파수, 위상 중 한가지 이상을 변화시키는 방식
    - 반송파를 이용하는 목적은 정보를 전자기파의 형태로 전송하거나, 또는 여러 주파수 대역에 반송파를 배분하는 주파수 분할 다중화를 통해 단일한 전송 매체를 공유하는 데에 있다.

### 디지털-대-아날로그 변환 유형

![Untitled 1](https://user-images.githubusercontent.com/91832324/233918033-f76160ed-b9ce-4115-879b-9d85fad8e324.png)

- 변조란 ?
    - 신호를(전송로 특성에 맞도록) 반송파(케리어)에 실어 정보 신호의 주파수를 높은쪽으로 옮기는 작업
- 변조의 종류
    
    
    | 전송신호 | 반송파 | 명칭 | 방법 |
    | --- | --- | --- | --- |
    | 아날로그 | 아날로그 | 연속파 아날로그 변조 | AM, FM, PM |
    | 아날로그 | 디지털 | 펄스 아날로그 변조 | PCM, DM |
    | 디지털 | 아날로그 | 연속파 디지털 변조 | ASK, PSK, FSK |
    | 디지털 | 디지털 | 펄스 디지털 변조 | Unipolar, Polar Bipolar |
- 신호율(보오율) : 초당 신호 단위의 수
    - S(신호율) = N ×    baud,
        - N : 데이터율(bps) - 초당 전송되는 비트의 수
            - r : 하나의 신호 요소에 전달되는 데이터 비트의 수

### 예제 5.1

> 아날로그 신호가 각 신호 요소에 4비트를 전송한다. 초당 1000개의 신호 요소가 보내진다면 비트율(bps)은 얼마인가?
> 

> **해답 :**
이 경우에는 r = 4, S = 1000이며 N 은 미지수이다. 다음과 같이 N을 구할 수 있다*.*
> 

### 예제 5.2

> 어떤 아날로그 신호의 비트율(bps=N)이 8000이고 보오율이  1000 보오(=S)이다. 각 신호 요소에는 몇 개의 데이터 요소를 전달하는가? 또한 몇 개의 신호 요소가 필요한가?
> 

> **해답 :**
이 경우에는 S = 1000, N = 8000이며 r과 L은 미지수이다. 다음과 같이 먼저 r의 값을 구하고 L의 값을 구한다.
> 

$S = N \times \frac{1}{r}$ **⇒** $r = \frac NS = \frac {8000}{1000} = 8 bits/baud$

$r = log_2L$ **⇒** $L = 2^r = 2^8 = 256$

## **진폭 편이 변조(ASK: Amplitude Shift Keying)**

- 진폭이 변하지만 주파수와 위상은 변하지 않는다
- 반송파의 진폭을 변화시켜 보내는 방식이다. 아날로그 변조 방식인 진폭 변조(AM)와 마찬가지로 이 방식은 다른 변조 방식에 비해 소음의 방해와 페이딩의 영향을 받기 쉽다
- ASK의 대역폭
    - B(대역폭) = (1 + d(0 ~1 사이의 값)) × S(신호율)
- **2진 ASK**
    
    ![Untitled 2](https://user-images.githubusercontent.com/91832324/233918037-e11fd920-b2fe-4aa6-a3cb-44f4d059198a.png)

    
    S(보오율) = N(=5(bps)) * 1/r(=1) =5 baud
    
    - ASK의 대역폭
        - B(대역폭) = (1 + d (0 과 1 사이의 값)) × S(신호율)
- **2진 ASK의 구현**
    
    ![Untitled 3](https://user-images.githubusercontent.com/91832324/233918040-47c4fe23-ec23-4333-90d0-5d767e2ccc4b.png)

    

### 예제 5.3

> 200 kHz에서부터 300 kHz에 걸치는 100 kHz의 대역을 사용할 수 있다. d = 1인 ASK를 사용하는 경우의 반송파의 주파수는 무엇인가?
> 

> **해답 :**
대역의 중간 지점은 250 kHz이다. 이는 반송파의 주파수 fc는  250 kHz인 것을 말한다. 비트율을 구하기 위해 다음 식을 사용할 수 있다.
> 

$B = (1 + d) \times S = 2 \times N \times \frac 1 r = 2 \times N = 100kHz$ **⇒ $N = 50kbps$**

### 예제 5.4

> 양방향으로 통신하기 위해 보통 전이중 링크를 사용한다. 그림 5.5와 같이 두 개의 반송파를 사용하고 2개 구간으로 대역을 나누어야 한다. 그림은 2 개 반송파의 위치와 대역폭을 보여주고 있다. 각 방향에 사용 가능한 대역폭은 이제 50 kHz이며, 각 방향 25 kbps의 데이터율을 제공한다.
**그림 5.5**
> 
> 
> ![Untitled 4](https://user-images.githubusercontent.com/91832324/233918044-663f6080-3851-42bc-bac1-2fbc34d42cda.png)
> 

## 주파수편이 변조
Frequency Shift Keying (FSK)

### BFSK(Binary FSK(**Frequency Shift Keying)**)

- 신호의 주파수가 2진 1 또는 0에 따라 변경
- 2진 주파수 편이 변조

![Untitled 5](https://user-images.githubusercontent.com/91832324/233918048-3286644d-f904-489f-b7ef-23d5e3fb0723.png)


- BFSK의 대역폭
    - B(Bandwidth) = (1 + d) × S + 2Δf
        - d : 0과 1사이
        - S : 신호율
        - 2Δf  : 요구 대역폭

![Untitled 6](https://user-images.githubusercontent.com/91832324/233918052-ec500d39-f8f5-4305-8cc5-68f23dc84090.png)

### 예제 5.5

> 100 kHz의 가용 대역이 영역 200 kHz부터 300 kHz에 걸쳐 있다. d =1인 FSK를 사용하여 데이터를 변조한다면 반송 주파수와 비트율은 얼마인가?
> 

> **해답 :**
이 문제는 예제 5.3과 유사하지만 FSK를 사용해야 한다. 중간점은 250 kHz이다. 2Δf를 50 kHz가 되도록 하자. 그런 경우에는 다음과 같다.
> 

$B = (1 + d) \times S + 2 \Delta f = 100$ **⇒ $2S = 50kHz \quad S = 25kbaud \quad N = 25kbps$**

### 예제 5.6

> 비트율 3 Mbps의 속도로 동시에 3비트를 보내야 한다. 반송파 주파수는 10 MHz이다. 준위의 개수(서로 다른 주파수의 개수)와 보오율과 대역폭을 구하라.
> 

> **해답 :**
L = 23 = 8이다. 보오율은 S = 3 MHz/3 = 1000 Mbaud이다. 이는 반송파 주파수는 서로 1 MHz씩 떨어져야 하는 (2∆f = 1 MHz)것을 의미한다. 대역폭은 B = 8 × 1000 = 8000이다. 그림 5.8에 각 주파수와 대역을 배당한 그림이 있다.
> 

![Untitled 7](https://user-images.githubusercontent.com/91832324/233918055-c9d5b03e-391d-43e1-8513-ddda70d4a8aa.png)

## 위상 편이 변조 (PSK: Phase Shift Keying)

- BPSK(Binary Phase Shift Keying)
    - 위상이 2진1 또는 0에 따라 변경
    - 2진 위상 편이 변조
    
    ![Untitled 8](https://user-images.githubusercontent.com/91832324/233917988-33628260-8c88-4bfe-8a95-aef1585ac5c1.png)

    
- QPSK(Quadrature Phase Shift Keying)
    - 각 신호 요소 마다 동시에 2비트를 사용할 수 있는 방법
    
    ![Untitled 9](https://user-images.githubusercontent.com/91832324/233917991-3f04ffae-d3f8-4024-9453-295164571a98.png)

    

![Untitled 10](https://user-images.githubusercontent.com/91832324/233917994-c775beec-fb3a-4571-a20c-31fd0c8821e2.png)


### 예제 5.7

> 12 Mbps로 전송하는 QPSK 신호의 대역폭을 구하라. d값은 0이다.
> 

> **해답 :**
QPSK에서는 한 신호요소가 2비트를 전송하므로 r = 2이다. 따라서 신호율(보오율) S = N × (1/r) = 6 Mbaud이다. d 값이  0 이므로 B = S = 6 Mbaud.
> 

### 성운 그림(Constellation diagram)

- 수평선 X축은 동위상 반송파
- 수직선 Y축은 구상 반송파
    
    ![Untitled 11](https://user-images.githubusercontent.com/91832324/233917997-cc7f390d-7f52-4a01-be23-d1fa72caae99.png)
    
- ASK(OOK), BPSK 그리고 QPSK 신호의 성운 그림

![Untitled 12](https://user-images.githubusercontent.com/91832324/233918001-3218e710-28e1-4b30-8aea-561d96a675b0.png)

## 구상 진폭 변조

**구상 진폭 변조(QAM)는 ASK와 PSK를 조합한 것이다.**

![Untitled 13](https://user-images.githubusercontent.com/91832324/233918003-cec001bf-9ba6-4770-a700-7d1c02a99526.png)

- ASK와 PSK의 조합
- 몇 가지 QAM의 성운 그림
- **16-QAM**
    
    ![Untitled 14](https://user-images.githubusercontent.com/91832324/233918005-d5908af1-0853-47ac-9b51-f88a270e29e5.png)
    
    ![Untitled 15](https://user-images.githubusercontent.com/91832324/233918009-08ac2837-e0d7-4a6e-aded-4fb9b2ecd3a2.png)

- **비트율과 보오율 비교**
    
    ![Untitled 16](https://user-images.githubusercontent.com/91832324/233918012-4258e7ed-867b-49b4-b95f-31b6485793fe.png)

### test

## 2. 아날로그-대-아날로그 변환

- 아날로그-대-아날로그 변환은 아날로그 신호로 아날로그 정보를 표현하는 것이다. 왜 이미 아날로그 신호인데 아날로그 신호를 보조 하는가 하는 의문을 가질 것이다. **매체가 특정 대역 통과 특성을 갖고 있거나 특정 대역만이 사용 가능한 경우에 변조가 필요하다.**
    
    ![Untitled 17](https://user-images.githubusercontent.com/91832324/233918014-033e50ea-9725-4fa0-80c0-347d6b65895f.png)

    
## **AM(Amplitude Modulation**)- 진폭 변조

- 신호의 진폭에 따라 반송파의 진폭이 변화한다.
    
    ![Untitled 18](https://user-images.githubusercontent.com/91832324/233918016-084040cb-029e-4625-85b3-de0207b231df.png)

- **AM 대역폭**
    
    > AM에 필요한 대역폭은 음성 신호의 대역폭으로부터 추론할 수 있다.
    $B_{AM} = 2B$
    > 
- **AM 라디오의 표준 대역 할당**
    - 오디오 신호(음성과 음악)의 대역폭은 5kHz
    - 각 방송국은 10kHz씩 할당
    - AM 대역 할당
    
    ![Untitled 19](https://user-images.githubusercontent.com/91832324/233918019-90d8d9ee-a3b3-4e38-b65b-35d9957e4e18.png)


## **FM(Frequency Modulation**)- 주파수 변조

- **변조 신호의 전압 준위 변화에 따라 반송 주파수가 변화**
    
    ![Untitled 20](https://user-images.githubusercontent.com/91832324/233918021-3f8b4ff2-c40f-4a31-955a-cbb3055425b8.png)

- **FM 대역폭**
    
    > FM의 전체 요구 대역폭은 음성 신호의 대역폭으로부터 다음과 같이 구할 수 있다.
    $B_{FM} = 2(1 + \beta) B$
    > 
- **FM 라디오의 표준 대역 할당**
    - 스테레오로 방송되는 오디오 신호 대역폭 15kHz
    - 각 FM 방송국은 최소 150kHz 대역폭 필요
    - 각 방송국에 200kHz(0.2MHz) 할당
    - FM 대역 할당
    
    ![Untitled 21](https://user-images.githubusercontent.com/91832324/233918025-e2a1d32c-846e-40f5-a07c-01e6f81904c3.png)


## **PM(Phase Modulation**)- 위상 변조

- 정보 신호의 진폭에 따라 반송파의 위상이 비례하여 변화한다.
    
    ![Untitled 22](https://user-images.githubusercontent.com/91832324/233918026-3053d659-2ae1-4c51-9c7e-d0675c83afef.png)

- **위상 변조 대역폭**
    
    > PM의 전체 요구 대역폭은 변조되는 신호의 대역폭과 최대 진폭으로부터 다음과 같이 구할 수 있다.
    $B_{PM} = 2(1 + \beta)B$
    >