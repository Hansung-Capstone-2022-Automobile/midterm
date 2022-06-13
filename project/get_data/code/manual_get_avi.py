import cv2
import RPi.GPIO as GPIO
import time
from make_avi import make_video
from time import sleep
import logging

# 모터 상태
STOP = 0
FORWARD = 1
BACKWORD = 2

# 모터 채널
CH1 = 0
CH2 = 1

# PIN 입출력 설정
OUTPUT = 1
INPUT = 0

# PIN 설정
HIGH = 1
LOW = 0

# 실제 핀 정의
# PWM PIN
ENA = 26  # 37 pin
ENB = 0  # 27 pin

# GPIO PIN
IN1 = 19  # 37 pin
IN2 = 13  # 35 pin
IN3 = 6  # 31 pin
IN4 = 5  # 29 pin


# 핀 설정 함수
def setPinConfig(EN, INA, INB):
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    # 100khz 로 PWM 동작 시킴
    pwm = GPIO.PWM(EN, 100)
    # 우선 PWM 멈춤.
    pwm.start(0)
    return pwm


# 모터 제어 함수
def setMotorContorl(pwm, INA, INB, speed, stat):
    # 모터 속도 제어 PWM
    pwm.ChangeDutyCycle(speed)

    # forward
    if stat == FORWARD:
        GPIO.output(INA, HIGH)
        GPIO.output(INB, LOW)

    # backward
    elif stat == BACKWORD:
        GPIO.output(INA, LOW)
        GPIO.output(INB, HIGH)

    # 정지
    elif stat == STOP:
        GPIO.output(INA, LOW)
        GPIO.output(INB, LOW)


# 모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
def setMotor(ch, speed, stat):
    if ch == CH1:
        # pwmA는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
        setMotorContorl(pwmA, IN1, IN2, speed, stat)
    else:
        # pwmB는 핀 설정 후 pwm 핸들을 리턴 받은 값이다.
        setMotorContorl(pwmB, IN3, IN4, speed, stat)

def forward():
    setMotor(CH1, 25, FORWARD)
    setMotor(CH2, 27, FORWARD)

def backward():
    setMotor(CH1, 20, BACKWORD)
    setMotor(CH2, 22, BACKWORD)

def left():
    setMotor(CH1, 15, BACKWORD)
    setMotor(CH2, 17, FORWARD)

def right():
    setMotor(CH1, 17, FORWARD)
    setMotor(CH2, 15, BACKWORD)

def stop():
    setMotor(CH1, 0, STOP)
    setMotor(CH2, 0, STOP)

# GPIO 모드 설정
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# 모터 핀 설정
# 핀 설정후 PWM 핸들 얻어옴
pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)

def make_video(video):
    cap = cv2.VideoCapture(video)
    logging.info('video start')
    # 열렸는지 확인
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    # 웹캠의 속성 값을 받아오기
    # 정수 형태로 변환하기 위해 round
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #w = 320
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #h = 240
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

    # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # 1프레임과 다음 프레임 사이의 간격 설정
    delay = round(1000/fps)

    # 웹캠으로 찰영한 영상을 저장하기
    # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
    # 가로세로 지정해야 할 경우 w:320, h:240
    out = cv2.VideoWriter('output2.avi', fourcc, 10, (w, h))

    # 제대로 열렸는지 확인
    if not out.isOpened():
        print('File open failed!')
        cap.release()
        sys.exit()

    # 프레임을 받아와서 저장하기
    while True:                 # 무한 루프
        ret, frame = cap.read() # 카메라의 ret, frame 값 받아오기

        if not ret:             #ret이 False면 중지
            break
        image = cv2.flip(frame, 1) # 영상 좌우반전
        out.write(image) # 영상 데이터만 저장. 소리는 X
        
        cv2.imshow('frame', image)
        
        keyValue = cv2.waitKey(delay)
        
        if keyValue == 27: # esc를 누르면 강제 종료
            break
        elif keyValue == 82:
            print("go")
            carState="go"
            forward()
        elif keyValue == 84:
            print("stop")
            carState="stop"
            stop()
        elif keyValue == 81:
            print("left")
            carState="left"
            left()
            time.sleep(0.3)
            forward()
        elif keyValue == 83:
            print("right")
            carState="right"
            right()
            time.sleep(0.3)
            forward()
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    make_video(0)
    GPIO.cleanup()
