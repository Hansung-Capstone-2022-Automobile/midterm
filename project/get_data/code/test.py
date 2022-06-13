import make_avi as video
import cv2
#video.make_video(0)


def test():
    global carState
    while True:
        keyValue = cv2.waitKey(10)
        if keyValue == ord('q'):
            pass
            #break
        elif keyValue == 82:
            print("go")
            carState = "go"
            #forward()
        elif keyValue == 84:
            print("stop")
            carState = "stop"
            #stop()
        elif keyValue == 81:
            print("left")
            carState = "left"
            #left()
            #time.sleep(0.3)
            #forward()
        elif keyValue == 83:
            print("right")
            carState = "right"
            #right()
            #time.sleep(0.3)
            #forward()
if __name__ == '__main__':
    test()
    print(carState)