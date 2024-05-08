import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone
import time

import pyzed.sl as sl

model=YOLO('yolov8s.pt')

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

err = zed.open(init_params)
if(err!=sl.ERROR_CODE.SUCCESS):
    exit(1)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

outputFile = "test_output.avi"

vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1020,500))

runtime_parameters = sl.RuntimeParameters()
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # A new image is available if grab() returns ERROR_CODE.SUCCESS
    i = 0
    image = sl.Mat()
    depth_zed = sl.Mat(zed.get_camera_information().camera_configuration.resolution.width, zed.get_camera_information().camera_configuration.resolution.height, sl.MAT_TYPE.F32_C1)
    runtime_parameters = sl.RuntimeParameters()

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n") 
    tracker=Tracker()

    area1=[(520,230),(850,260),(855,334), (510,310)]
    area2=[(510,335),(860,350),(856,430), (503,420)]

    people_enter={}
    counter1=[]

    people_exit={}
    counter2=[]

    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns ERROR_CODE.SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image

            frame = image.get_data()[:, :, :3]

            # print("Frame shape: ", frame.shape)
            frame=cv2.resize(frame,(1020,500))

            results=model.predict(frame)
            #   print(results)
            a=results[0].boxes.boxes
            a = a.cpu()
            px=pd.DataFrame(a).astype("float")
        #    print(px)
            list=[]         
            for index,row in px.iterrows():
        #        print(row)
        
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=class_list[d]
                if 'person' in c:
                    print("I am here")
                    list.append([x1,y1,x2,y2])

                bbox_id=tracker.update(list)
                for bbox in bbox_id:
                    x3,y3,x4,y4,id=bbox

                    # print(area1, (x4, y4))
                    # print(area2, (x4, y4))

                    # results0 = 0
                    # results1 = 0
                    # results2 = 0
                    # results3 = 0

                    results0 = cv2.pointPolygonTest(np.array(area1,np.int32), (x4, y4), False)
                    # area1=[(520,230),(850,260),(855,334), (510,310)]
                    # area2=[(510,335),(860,350),(856,430), (503,420)]
                    # if x4>=max(area1[0][0], area1[3][0]) and x4<=min(area1[1][0], area1[2][0]) and y4>=min(area1[0][1], area1[3][1]) and y4<=max(area1[1][1], area1[2][1]):
                    #     results0 = 1
                    if results0>0:
                        # print("exit 1")
                        people_exit[id] = (x4, y4)
                    if id in people_exit:
                        results1 = cv2.pointPolygonTest(np.array(area2,np.int32), (x4, y4), False)
                        # if x4>=max(area2[0][0], area2[3][0]) and x4<=min(area2[1][0], area2[2][0]) and y4>=min(area2[0][1], area2[3][1]) and y4<=max(area2[1][1], area2[2][1]):
                        #     results1 = 1
                        if results1>0:
                            # print("exit 2")
                            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
                            cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
                            cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)                           
                            if counter2.count(id)==0:
                                counter2.append(id)


                    results2 = cv2.pointPolygonTest(np.array(area2,np.int32), (x4, y4), False)
                    # if x4>=max(area2[0][0], area2[3][0]) and x4<=min(area2[1][0], area2[2][0]) and y4>=min(area2[0][1], area2[3][1]) and y4<=max(area2[1][1], area2[2][1]):
                    #     results2 = 1
                    if results2>0:
                        # print("entry 1")
                        people_enter[id] = (x4, y4)
                    if id in people_enter:
                        results3 = cv2.pointPolygonTest(np.array(area1,np.int32), (x4, y4), False)
                        # if x4>=max(area1[0][0], area1[3][0]) and x4<=min(area1[1][0], area1[2][0]) and y4>=min(area1[0][1], area1[3][1]) and y4<=max(area1[1][1], area1[2][1]):
                        #         results3 = 1
                        if results3>0:                            
                            # print("entry2")
                            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
                            cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
                            cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                            if counter1.count(id)==0:
                                counter1.append(id)  

                    print(results0, results2)          
                
                cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),1)
                cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),1) 

                print(people_exit) 
                print(people_enter)
            
                ex = len(counter2)
                en = len(counter1)

                cvzone.putTextRect(frame, f'Entered: {en}', (50, 50), 0.8, 2, (255, 255, 255), (0, 0, 0), cv2.FONT_HERSHEY_COMPLEX)
                cvzone.putTextRect(frame, f'Exited: {ex}', (50, 100), 0.8, 2, (255, 255, 255), (0, 0, 0), cv2.FONT_HERSHEY_COMPLEX)

                cv2.imshow("RGB", frame)
                vid_writer.write(frame.astype(np.uint8))
                if cv2.waitKey(1)&0xFF==27:
                    break
            

            # plt.imshow(image.get_data())
            # plt.show()
            # cv2.imshow("Image", image.get_data())
            # cv2.waitKey(1)

            # zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            # depth_ocv = depth_zed.get_data()
            # print(depth_ocv[int(len(depth_ocv)/2)][int(len(depth_ocv[0])/2)])
            # cv2.imshow("Image Depth", depth_ocv)

            # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
            # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(), timestamp.get_milliseconds()))
        

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

# cap=cv2.VideoCapture('shoppingmall.mp4')


# my_file = open("coco.txt", "r")
# data = my_file.read()
# class_list = data.split("\n") 
# #print(class_list)

# count=0

# tracker=Tracker()

# area1=[(708,239),(690,253),(945,334),(959,317)]
# area2=[(681,257),(677,265),(927,353),(937,342)]

# people_enter={}
# counter1=[]

# people_exit={}
# counter2=[]
# while True:    
#     ret,frame = cap.read()
#     if not ret:
#         break
#     count += 1
#     if count % 3 != 0:
#         continue
#     frame=cv2.resize(frame,(1020,500))
   

#     results=model.predict(frame)
#  #   print(results)
#     a=results[0].boxes.boxes
#     a = a.cpu()
#     px=pd.DataFrame(a).astype("float")
# #    print(px)
#     list=[]         
#     for index,row in px.iterrows():
# #        print(row)
 
#         x1=int(row[0])
#         y1=int(row[1])
#         x2=int(row[2])
#         y2=int(row[3])
#         d=int(row[5])
#         c=class_list[d]
#         if 'person' in c:
#             list.append([x1,y1,x2,y2])
#     bbox_id=tracker.update(list)
#     for bbox in bbox_id:
#         x3,y3,x4,y4,id=bbox
#         results0 = cv2.pointPolygonTest(np.array(area1,np.int32), (x4, y4), False)
#         if results0>0:
#             people_exit[id] = (x4, y4)
#         if id in people_exit:
#             results1 = cv2.pointPolygonTest(np.array(area2,np.int32), (x4, y4), False)
#             if results1>0:
#                 cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
#                 cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
#                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
#                 if counter2.count(id)==0:
#                     counter2.append(id)


#         results2 = cv2.pointPolygonTest(np.array(area2,np.int32), (x4, y4), False)
#         if results2>0:
#             people_enter[id] = (x4, y4)
#         if id in people_enter:
#             results3 = cv2.pointPolygonTest(np.array(area1,np.int32), (x4, y4), False)
#             if results3>0:
#                 cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
#                 cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
#                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
#                 if counter1.count(id)==0:
#                     counter1.append(id)            
        
#     cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),1)
#     cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),1)  
 
#     ex = len(counter2)
#     en = len(counter1)

#     cvzone.putTextRect(frame, f'Entered: {en}', (50, 50), 0.8, 2, (255, 255, 255), (0, 0, 0), cv2.FONT_HERSHEY_COMPLEX)
#     cvzone.putTextRect(frame, f'Exited: {ex}', (50, 100), 0.8, 2, (255, 255, 255), (0, 0, 0), cv2.FONT_HERSHEY_COMPLEX)
   
   

 
#     cv2.imshow("RGB", frame)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cap.release()
# cv2.destroyAllWindows()

