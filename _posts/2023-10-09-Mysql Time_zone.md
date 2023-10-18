---
layout: single

title: "MySQL Time Zone"

date: 2023-10-10 20:00:00 +0900
lastmod: 2023-10-10 20:00:00 +0900 # sitemap.xml에서 사용됨

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
# MySQL TimeZone 설정방법

DB에 들어가는 시간이 한국 시간으로 들어가야 나중에 DB관리할때 편하기 때문에 하는게 좋음

나중에 대형 프로젝트를 하게 될 경우 타 나라에 사는 사람들과 프로젝트를 하는 경우 DB데이터를 넣는 시간을 자기에 맞게 해야지 편리함

[MySQL :: Time zone description tables](https://dev.mysql.com/downloads/timezones.html)

![](\assets\image\Project\Time_zone/Untitled.png)

위 사이트에 들어가서 빨간색 네모 다운

- POSIX란?
    
    **Portable Operating System Interface**약자로 **이식 가능 운영 체제 인터페이스**라는 뜻을 가진다.
    
    다양한 운영 체제에서 동일한 방식을 프로그램을 작성, 실행할 수 있도록 해준다.
    

다운 받은 후 zip을 풀어준다.

`MySQL 8.0 Command Line Client`라는 프로그램을 실행하고 자신이 타임존을 설정할 DB 비밀번호를 입력해준다.

이후 **`select** @@global.time_zone, @@session.time_zone;` 명령어를 쳐준다.

![](\assets\image\Project\Time_zone/Untitled 1.png)

위와 같은 화면이 뜬다면 추가 설정이 필요하다.

`use mysql` 명령어를 치고 `source` + `zip을 해제한 폴더` + `sql파일 이름`을 쳐준다.

예시) 

![](\assets\image\Project\Time_zone/Untitled 2.png)

그러면 Query OK와 같은 명령어가 쫙 뜨면 성공이다.

![](\assets\image\Project\Time_zone/Untitled 3.png)

```sql
SET GLOBAL time_zone='Asia/Seoul';

SET time_zone='Asia/Seoul';
```

이후 위 명령어를 입력해주면 완료

다시 **`select** @@global.time_zone, @@session.time_zone;`명령어를 치면 `Asia/Seoul`이 적용된게 확인된다.

![](\assets\image\Project\Time_zone/Untitled 4.png)