---
layout: single

title: "컴퓨터 그래픽스의 개요"

date: 2023-09-07 20:00:00 +0900
lastmod: 2023-09-07 20:00:00 +0900 # sitemap.xml에서 사용됨

author_profile: true

header:
  overlay_image: 

  overlay_filter: 0.5 # 투명도

categories: 
  - College Graphics

tags: 
    - Computer Graphics
    - College

# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "table of content" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---
# 01 컴퓨터 그래픽스의 개요

### 컴퓨터그래픽 정의

- 컴퓨터 처리에 속하는 수치, 문자를 입력 데이터로써 도형, 화상 정보를 출력, 즉 **화상을 생성 처리하는 일**

## 컴퓨터 그래픽스의 발전

- 1970년대
    - 60년대 나온 **CAD 시스템이 널리 사용**되기 시작
- 1980년대
    - DOS 시스템에서 Window로 넘어감
    - Graphics Art
- 1990년대
    - 하드웨어 분야에서 급격한 발전이 이루어진 시기
    - 3D Graphics 발전
    - 하드웨어 발전이 소프트웨어 발전을 이끌어 줌
        - H/W가 발전할수록 S/W의 성능을 100% 이끌 수 있기 때문
        - 새 H/W가 나온다면 이를 활용해 더 좋은 어플을 개발이 가능하기 때문
- 2000년대
    - 실시간 렌더링으로 인해 물체의 사실감, 자연스러움 증가시키는 그래픽 기술
    ⇒ 진짜 물건이나 사람 대신 그래픽으로 처리 가능
    - 모바일 환경 및 무선 환경으로 PDA, 휴대폰 등

## 2차원 그래픽과 3차원 그래픽

### 2차원 그래픽

- 벡터 그래픽$$_{Vector \ Graphics}$$
    - 그래픽에 사용된 객체들을 수학적 함수로 표현해 기억 공간에 저장하는 방식
    - **화면을 확대하더라도 화질 변화 X**
    
    ```bash
    <?xml version = "1.0"?>
    <svgwidth="200" height="200">
         <ellipsecx="110" cy="50" rx="70" ry="40">
             style="fill:blue;stroke:blue; stroke-width:2"/
         <polygonstyle="fill:green;stroke:green;"
     stroke-width:2"points="130,40 140,190 50,190"
          />
    </svg>
    ```
    
    SVG에 위와 같이 입력 시 아래와 같은 그림이 출력이 된다.
    
    ![](\assets\image\ComputerGraphics\01/Untitled.png)
    
- 래스터 그래픽$$_{Raster \ Graphics}$$
    - 래스터 그래픽 출력 장치에 표시하기 위한 그래픽 데이터를 픽셀 단위로 기억 공간에 저장
    - **화면 확대 시 화질 감소**
    - **화질이 출력 장치의 해상도 성능에 따라 바뀜**
        
        ![](\assets\image\ComputerGraphics\01/Untitled 1.png)
        
- 래스터 그래픽과 벡터 그래픽의 차이
    
    ![](\assets\image\ComputerGraphics\01/Untitled 2.png)
    
    - 위 그림을 보면 비트맵은 화면 확대 시 화질이 깨지지만 벡터는 깨지지 않는다는 것을 확인할 수 있다

### 3차원 그래픽

- 3차원 그래픽 생성과정
    1. 모델링$$_{Modeling}$$
        - 3차원 좌표계에서 물체의 모양을 표현하는 과정
        - 와이어프레임$$_{Wireframe}$$ 모델
        - 다각형 표면$$_{Polygon \ Surface}$$ 모델
        - 솔리드$$_{Solid}$$ 모델링
        - 3차원 스캔에 의한 모델링
        
        ---
        
        ![](\assets\image\ComputerGraphics\01/Untitled 3.png)
        
    2. 투영$$_{Projection}$$
        - 3차원 객체를 2차원 화면에 투영
        - 즉, 3차원 객체를 사진처럼 2차원으로 축소 시키는 것이다.
        - 평형 투영법과 원근 투영법
    3. 렌더링$$_{Rendering}$$
        - 색상과 명암의 변화와 같은 3차원적인 질감을 더하여 현실감을 추가하는 과정
        - 은면의 제거$$_{Hidden \ Surface \ Removal}$$
            - 3D 장면에서 물체 중 일부가 다른 물체에 의해 가려져 숨겨진 부분을 결정하고 표시하지 않는 기술
        - 쉐이딩$$_{Shading}$$, 텍스쳐 매핑$$_{Texture \ Mapping}$$ , 그림자$$_{Shadow}$$
            - 쉐이딩은 3D 모델의 표면에 빛과 그림자의 효과를 적용하여 모델이 입체적으로 보이도록 만드는 기술
            - 텍스쳐 매핑은 3D 모델의 표면에 2D 이미지나 패턴을 적용해 모델에 표면에 목재, 벽돌, 피부, 옷감과 같은 다양한 재질과 디자인을 부여하는 기술
            - 그림자 기술은 빛의 방향과 위치에 따른 물체의 그림자를 생성하고 표시하는 기술입
        - 광선추적법$$_{Ray \ Tracing}$$
            - 광선추적법은 3D 그래픽스 렌더링에서 현실적인 빛의 반사, 굴절, 그림자, 반사 등을 물리적으로 정확하게 시뮬레이션하는 렌더링 기술
            
            ![](\assets\image\ComputerGraphics\01/Untitled 4.png)
            
            ![](\assets\image\ComputerGraphics\01/Untitled 5.png)
            

## 그래픽스, 이미지처리, 애니메이션 및 가상현실

### 그래픽스와 이미지처리

- 로젠펠드는 그래픽스, 이미지처리 및 컴퓨터 비전의 차이점을 구분

![](\assets\image\ComputerGraphics\01/Untitled 6.png)

### 애니메이션과 가상현실

- 애니메이션$$_{Animation}$$
    - 일련의 정지된 그림이나 이미지를 연속적으로(초당 15 프레임 이상) 보여주어 연속된 동작으로 느낌
    - 인간의 잔상효과를 이용
        
        ![](\assets\image\ComputerGraphics\01/Untitled.gif)
        
- 가상현실$$_{Virtual \ Reality}$$
    - 컴퓨터 그래픽스 기술을 이용하여 가상공간과 객체들을 구축, 관찰자$$_{Viewer}$$가 가상공간을 돌아다니며 체험
    - 가상공간과 물체의 실시간 디스플레이(초당 15 Frame이상)가 중요

## 컴퓨터 그래픽스의 활용

### 컴퓨터 그래픽스의 활용분야

1. CAD
    - 건축 설계
    - 부품설계 및 도면작성$$_{Drafting}$$, 기계설계
2. 컴퓨터 애니메이션과 시뮬레이션
    - 프레임들의 빠른 연속적인 디스플레이
    - 교육, 훈련(조종 훈련), 물리학적 시스템의 특성과 동작 이해
3. 컴퓨터 디자인 및 아트
    - 상업 디자인
    - 창작 미술
    - 손상된 그림 복구
4. 게임 및 엔터테이먼트
    - 영화, 게임 , 뮤직 비디오, TV 프로그램 등
    - 컴퓨터 게임: 2차원 게임에서 3차원 게임으로 발전
5. 프리젠테이션 및 데이터 시각화$$_{Data \ Visualization}$$
    - 프리젠테이션 그래픽스$$_{Presentation \ Graphics}$$: 그래프, 차트, 비즈니스 그래픽, 프로젝트 관리
    - 컴퓨터 생성 모델$$_{Visualization}$$: 물리적, 금융, 경제 모델
    - 일기예보도 여기 포함
6. 멀티미디어 분야에서 활용
    - 그래픽은 멀티미디어 응용에서 가장 자주 이용하는 매체
    - 웹페이지, 디지털 방송, 휴대폰, 사이버 클래스, 가상환경의 구축, 아바타 생성
    - VR Chat도 여기 들어감
7. GUI
    - window
    - Icons
    - Menu
8. 전자 출판
    - 문서 준비 시스템 (Document Preparation System)
    - 출판 (DTP: 데스크톱 퍼블리싱)
9. 공간 정보의 표현
    - 지리 정보 시스템(GIS: Geographic Information System)
    - 차량 주행 안내 시스템(Car Navigation System)
    - C3I, C4I (Military, Police, Emergency monitoring) : 50년대 중반 SAGE 방공 시스템이 시초
10.  이미지 처리(Image Processing)
    - Feature Detection
    - Pattern Recognition
    - 3D Reconstruction(예: MRI, CT)