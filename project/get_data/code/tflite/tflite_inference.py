import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import sys

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

def details(model):
    # 모델 details
    for item in model.get_tensor_details():
        for key in item.keys():
            print("%s : %s" % (key, item[key]))
        print("")

def tf_inference(image):
    model_path = "C:/Users/User/Desktop/get_data/code/model/converted_model.tflite"
    model = tflite.Interpreter(model_path)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.allocate_tensors()

    #img = cv2.imread(image)
    #print(f'image:{image.shape}')
    #img = image
    #new_img = cv2.resize(image,(200,66))
    #new_img = new_img.astype(np.float32)
    new_img = image.astype(np.float32)
    #new_img /= 255.
    #new_img = np.expand_dims(new_img, axis=0)

    #model.set_tensor(input_details[0]['index'], new_img)
    #print(f"input_detail{input_details}")
    #print(f'new_img:{image.shape}')
    model.set_tensor(input_details[0]['index'], new_img)
    #print(input_details[0]['index']) # new_img와 같은 shape이어야 한다.
    model.invoke()

    predict_steering = model.get_tensor(output_details[0]['index'])

    return predict_steering

if __name__ == '__main__':
    #model_path = str("C:/Users/User/Desktop/get_data/code/model/converted_model.tflite")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()
    #cap.set(3, 1280)
    #cap.set(4, 720)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #print(frame.shape)
        imgs = cv2.flip(frame, 1)
        #cv2.imshow('frame',image)
        #print(imgs.shape)

        pre_img = img_preprocess(imgs)
        #print(f'pre_img:{pre_img.shape}')
        cv2.imshow("frame", pre_img)
        X = np.asarray([pre_img])
        #print(X)

        vector = np.vectorize(np.float64)
        predict_steering_angle = tf_inference(X)
        print(f'pred_steering_angle : {predict_steering_angle}')
        #print(predict_steering_angle)

        cv2.imshow('a',imgs)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



