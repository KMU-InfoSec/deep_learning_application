# IDA

## 사용법
1. IDA Pro 를 설치한다
2. IDA Pro 를 방화벽 규칙에 추가 한다. ( 외부로 가는 것 차단 )
3. [settings.py](./settings.py) 에 있는 다음 변수 값을 수정 한다.
*  `CPU_COUNT`, 'IDA_PATH', 'BASE_PATH'
4. [나스](http://203.246.112.134:5000/)에서 악성코드(/homes/virussign)를 받아서 [settings.py](./settings.py) 의 'ZIP_FILE_PATH'에 넣는다.
5. [나스](http://203.246.112.134:5000/)에서 idb를 다운로드한다.
6. [unzip_virussign.py](./unzip_virussign.py)를 실행 시킨다. (악성코드 압축 해제 -> 'INPUT_FILE_PATH'에 저장해줌)
7. 실험환경 디렉토리를 잘 생성해준다.
8. [main.py](./main.py)를 "관리자 권한으로" 실행 시킨다. (pycharm 관리자 권한, python 관리자 권한 둘 다 가능)


## 주의사항
1. 실험환경 디렉토리 확인 및 잘 맞추기
2. 용량이 부족할 수 있으므로 날짜를 적게 해서 자주 올리는걸 추천. (새벽 시간에도 활용)
3. idb, ops만 생성해야 하고, 완료 후에는 NAS(134)에 upload (/homes/idb 지우고 업로드, /homes/ops 지우고 업로드)
4. idb가 NAS에 없는 날짜: 9/1,9/2,10/13,10/14,10/15,10/29
5. 매일 어디까지 되있는지 상황보고하기
