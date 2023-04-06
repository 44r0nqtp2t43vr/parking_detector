from imageai.Detection import ObjectDetection
import cv2
import os

def get_center(box_points):
    return (int((box_points[0] + box_points[2]) / 2), int((box_points[1] + box_points[3]) / 2))

def draw_points(frame, points):
    for point in points:
        frame = cv2.rectangle(frame, point, (point[0] + 2, point[1] + 2), (0, 255, 0), 1)
    return frame

execution_path = os.getcwd()

car_detector = ObjectDetection()
car_detector.setModelTypeAsYOLOv3()
car_detector.setModelPath(os.path.join(execution_path, "objdet_model/yolov3.pt"))
car_detector.loadModel()

custom = car_detector.CustomObjects(bicycle=True, car=True, bus=True, truck=True)

video = cv2.VideoCapture('videos/parking_lot_1.mp4')

while (video.isOpened()):
    ret, frame = video.read()

    if ret == True:
        returned_image, detections = car_detector.detectObjectsFromImage(custom_objects=custom, input_image=frame, output_type="array", minimum_percentage_probability=30)
        center_list = [get_center(obj['box_points']) for obj in detections]
        returned_image = draw_points(returned_image, center_list)
        cv2.imshow('Video', returned_image)

        key = cv2.waitKey(20)
        if key == 27:
            break
    else:
        break

video.release()
cv2.destroyAllWindows()