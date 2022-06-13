import cv2
import RPi.GPIO as GPIO
import time
from time import sleep

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
    setMotor(CH1, 20, FORWARD)
    setMotor(CH2, 25, FORWARD)

def backward():
    setMotor(CH1, 20, BACKWORD)
    setMotor(CH2, 25, BACKWORD)

def left():
    setMotor(CH1, 15, BACKWORD)
    setMotor(CH2, 15, FORWARD)

def right():
    setMotor(CH1, 15, FORWARD)
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


def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    
    filepath = "/home/pi/Desktop/get_data/ai_car/img"
    i = 0
    carState = "stop"
    while (camera.isOpened()):

        keyValue = cv2.waitKey(10)

        if keyValue == ord('q'):
            break
        elif keyValue == 82:
            print("go")
            carState = "go"
            forward()
        elif keyValue == 84:
            print("stop")
            carState = "stop"
            stop()
        elif keyValue == 81:
            print("left")
            carState = "left"
            left()
        elif keyValue == 83:
            print("right")
            carState = "right"
            right()

        _, image = camera.read()
        #image = cv2.flip(image, -1)
        cv2.imshow('Original', image)

        height, _, _ = image.shape
        save_image = image[int(height / 2):, :, :]
        save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
        save_image = cv2.GaussianBlur(save_image, (3, 3), 0)
        save_image = cv2.resize(save_image, (200, 66))
        cv2.imshow('Save', save_image)

        if carState == "left":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 45), save_image)
            i += 1
        elif carState == "right":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 135), save_image)
            i += 1
        elif carState == "go":
            cv2.imwrite("%s_%05d_%03d.png" % (filepath, i, 90), save_image)
            i += 1

    cv2.destroyAllWindows()

# 제어 시작
if __name__ == '__main__':
    main()
    GPIO.cleanup()
