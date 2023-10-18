---
layout: single

title: "MySQL Download"

date: 2023-10-09 20:00:00 +0900
lastmod: 2023-10-09 20:00:00 +0900 # sitemap.xml에서 사용됨

author_profile: true

header:
  overlay_image: 

  overlay_filter: 0.5 # 투명도

categories: 
  - College Project

tags: 
    - Project
    - College

# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "table of content" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---
# MYSQL 설치 방법

[MySQL :: Download MySQL Installer](https://dev.mysql.com/downloads/windows/installer/8.0.html)

위 사이트에서 version 8.0.34버전 다운

![](\assets\image\Project\Mysql/Untitled.png)

- 빨간색 부분 누르면 로그인 없이 다운 가능
    
    ![](\assets\image\Project\Mysql/Untitled 1.png)
    
- Custom으로 설치를 진행
    - Server only는 주로 Test코드를 실행하는 용도로 많이 사용
    - Client only는 이미 MySQL DB 서버가 다른 장소에 설치되어있으면 사용
    - Full은 전부다 설치해줌
    - Custom은 원하는대로 골라서 설치 가능
- Full로 설치하면 자원 리소스를 너무 많이 잡아먹어서 Custom으로 진행
    
    ![](\assets\image\Project\Mysql/Untitled 2.png)
    
- 아래 4개를 추가해주고 진행
    1. MySQL Servers - MySQL Server - MySQL Server 8.0 - MySQL Server 8.0.26
    2. Applications - MySQL Workbench - MySQL Workbench 8.0 - MySQL Workbench 8.0.26
    3. MySQL Connectors - Connectors/J - Connectors/J 8.0 - Connectors/J 8.0.26
    4. Documentation - Samples and Examples - Samples and Examples 8.0 - Samples and Examples 8.0.26
    
    ![](\assets\image\Project\Mysql/Untitled 3.png)
    
- 이후 Excute를 누르면 Progress에 실시간 다운%가 보인다
- 다운 다 됐는데 왼쪽에 빨간색 느낌표가 뜬다면 Back을 눌렀다 다시 Excute를 눌러서 설치 재 실행
    
    ![](\assets\image\Project\Mysql/Untitled 4.png)
    
- 성공적으로 설치가 됐다면 아래 2개 항목의 환경 설정이 필요하다고 나온다.
    
    ![](\assets\image\Project\Mysql/Untitled 5.png)
    
- 아래 화면과 동일하게 세팅해주면 된다.
    
    ![](\assets\image\Project\Mysql/Untitled 6.png)
    
- **Config Type을 Dev로 무조건 해줘야한다.**
- **만약 Port 3306에 포트 충돌이 일어난다면 이미 해당 포트를 사용하는 프로그램이 있다는것!**
- 만약 본인이 사용자명이 한글이 포함되었다면 빨간색 박스를 클릭해서 추가 설정을 해줘야 한다.
    - 
    
    ![](\assets\image\Project\Mysql/Untitled 7.png)
    
- 8.0 이상 버전을 사용하니까 첫 번째로 사용
    
    ![](\assets\image\Project\Mysql/Untitled 8.png)
    
- 비밀번호는 기억하기 쉬운 0000으로 진행
- 아마 비밀번호는 개인마다 달라도 되는걸로 기억하는데….
- Add User는 할 필요 없음 최초 설치시에 관리자 Root의 비번으로 들어가기 때문
    
    ![](\assets\image\Project\Mysql/Untitled 9.png)
    
- 이름만 MySQL로 바꾸면 됨
    
    ![](\assets\image\Project\Mysql/Untitled 10.png)
    
- 맨 위는 Windows 서비스를 실행하는 사용자 및 관리자 그룹에 전체 액세스 권한을 부여
즉, 보안 강화를 해준다고 보면 된다.
대신 DB파일 직접 접근할땐 조심해야 한다는 단점이 있다.
- 2번째는 액세스 권한을 직접 부여 및 설정이 가능하다.
이건 할줄 몰라서 ㅎㅎ;;
- 맨 아래는 서버를 만들고 권한을 추후에 관리함
즉, Server를 만들고 난 뒤 서버마다 권한을 개별적으로 관리한다는 뜻
- 별 차이들을 못 느껴서 1번째걸로 진행
    
    ![](\assets\image\Project\Mysql/Untitled 11.png)
    
- 이후 Excute를 누르면 왼쪽 동그라미에 녹색 체크가 뜨면서 완료되는게 확인된다.
- **만약 완료가 안된 경우에는 사용자명이 한글일 가능성이 농후하니 확인할 것**
    
    ![](\assets\image\Project\Mysql/Untitled 12.png)
    
- Server 설정은 끝났고 이제 Sample 설정을 해야한다.
- 아까 Root에 비번을 지정한걸 입력하고 Check를 누르면 Next버튼이 활성화 된다.
본인은 0000임
    
    ![](\assets\image\Project\Mysql/Untitled 13.png)
    
- Excute를 누르면 끝
    
    ![](\assets\image\Project\Mysql/Untitled 14.png)
    
- Status에 완료되었다고 뜨면 성공한 것
- 이후 Start MySQL Workbench after Setup을 해제하고 Finish를 누르면 됨