from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO("D:\\GitHub\\Face-Eyes_Detection\\PBG_modelBEST.pt")

# load video
# video_path = 'D:\\GitHub\\Face-Eyes_Detection\\src\\test.mp4'
# cap_ecr = cv2.VideoCapture(video_path)
cap_ecr=cv2.VideoCapture(0)
cap_ecr.set(3, 840)
cap_ecr.set(4, 680)
font=cv2.FONT_HERSHEY_COMPLEX

ret = True
# read frames
while ret:
    ret, frame = cap_ecr.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break



cap_ecr.release()
cv2.destroyAllWindows()

# ###############################################
#################
#################
#################################################
# from ultralytics import YOLO
# import cv2

# # load yolov8 model
# model = YOLO("D:\\GitHub\\Face-Eyes_Detection\\PBG_modelBEST.pt")

# # load video
# # video_path = 'D:\\GitHub\\Face-Eyes_Detection\\src\\test.mp4'
# # cap_ecr = cv2.VideoCapture(video_path)
# cap_ecr = cv2.VideoCapture(0)
# cap_ecr.set(3, 840)
# cap_ecr.set(4, 680)
# font = cv2.FONT_HERSHEY_COMPLEX

# ret = True
# # read frames
# while ret:
#     ret, frame = cap_ecr.read()

#     if ret:
#         # detect objects
#         results = model.predict(frame, conf=0.7)  

#         # track objects
#         results = model.track(frame, persist=True)

#         # plot results
#         # cv2.rectangle
#         # cv2.putText
#         frame_ = results[0].plot()

#         # visualize
#         cv2.imshow('frame', frame_)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

# cap_ecr.release()
cv2.destroyAllWindows()
