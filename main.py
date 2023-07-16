import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np


def sort_array_func(val):
    return val[3]

st.set_page_config(page_title="Beads_Counting",page_icon="💫")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Beads Counting App")

st.subheader("Select the yourn choice")

model_choice = st.selectbox("select",["best_cpu_beads_model_yolov8","updated_beads","yolov8_muthu_best"])
choice = st.selectbox("select",["Upload image","Upload Video","Real Time"])
conf = st.number_input("conf",0.0,0.9)

if choice == "Upload image":
    
    image_data = st.file_uploader("Upload the Image")
    img_summit_button = st.button("Summit")
    
    if img_summit_button:
        
        model = YOLO("weights/"+ model_choice +".pt")   
        classes_list = model.names
        detected_object_name_list = []
        box_count_num = 0

        image = Image.open(image_data)
        image.save("input_data_image.png")
        frame = cv2.imread("input_data_image.png")
                
        # model prediction
        results = model.predict(source=frame,iou=0.7,conf=conf)
        
        get_array = results[0].boxes.numpy().boxes.tolist()

        # function to sort array 
        get_array.sort(key=sort_array_func)
        
        #------------------------------------------- processing the frame some condition ----------------------------------------#
        if len(get_array) == 0:
            pass
        else:
            for ind,i in enumerate(get_array):

                    # overlap condition 
                    if len(get_array) != ind+1 :
                        
                        # print(ind+1," ==> ",int((int(get_array[ind+1][1])-i[1])))
                        if int((int(get_array[ind+1][1])-i[1])) > 12:
                            
                        # if 1==1:
                            last = classes_list[round(i[-1])]
                            detected_object_name_list.append(last)
                            box_count_num += 1
                            cv2.rectangle(frame,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(0, 255, 0), 2)
                            cv2.putText(frame,str(box_count_num),(int(i[0])-70,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2,cv2.LINE_AA) 
                            cv2.line(frame,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)
                            
                    else:
                        
                        last = classes_list[round(i[-1])]
                        detected_object_name_list.append(last)
                        box_count_num += 1
                        
                        cv2.rectangle(frame,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(0, 255, 0), 2)
                        cv2.putText(frame,str(box_count_num),(int(i[0])-40,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv2.LINE_AA) 
                        cv2.line(frame,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)

        cv2.putText(frame,"Count => "  + str(len(detected_object_name_list)),(10,680),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) 

        box_count_num = 0
        detected_object_name_list = []

        st.image(frame)


if choice == "Upload Video":

    video_data = st.file_uploader("Upload the Video", type = ['mp4'])
    
    video_summit_button = st.button("Summit!")
    
    if video_summit_button:
        
    # to load the yolo weights file using torch hub
        model = YOLO("weights/"+ model_choice +".pt")   
        
        classes_list = model.names
        detected_object_name_list = []
        box_count_num = 0
        count  = ""

        def sort_array_func(val):
            return val[3]

        frame_num = 5
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_data.read())
        video = cv2.VideoCapture(tfile.name)
        # video = cv2.VideoCapture(0)
        
        img_show = st.image([])
        j = st.button("STOP")
        
        
        while True:   
            _,frame = video.read()
            # frame_num += 1
            # model prediction
            if frame_num == 5:
        
                # frame_num = 0
                # model prediction
                results = model.predict(source=frame,iou=0.7,conf=conf)
                # cv2.imshow("YOLOv8 Inference", results[0].plot())
                get_array = results[0].boxes.numpy().boxes.tolist()

                # function to sort array 
                get_array.sort(key=sort_array_func)

      
                #------------------------------------------- processing the frame some condition ----------------------------------------#
                if len(get_array) == 0:
                    pass
                else:
                    
                    for ind,i in enumerate(get_array):
                            # overlap condition 
                            if len(get_array) != ind+1 :
                                
                                # print(ind+1," ==> ",int((int(get_array[ind+1][1])-i[1])))
                                if int((int(get_array[ind+1][1])-i[1])) > 12:
                                    
                                    last = classes_list[round(i[-1])]
                                    detected_object_name_list.append(last)
                                    box_count_num += 1
                                    cv2.rectangle(frame,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(0, 255, 0), 2)
                                    cv2.putText(frame,str(box_count_num),(int(i[0])-70,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2,cv2.LINE_AA) 
                                    cv2.line(frame,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)
                            
                            else:
                                
                                last = classes_list[round(i[-1])]
                                detected_object_name_list.append(last)
                                box_count_num += 1
                                
                                cv2.rectangle(frame,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(0, 255, 0), 2)
                                cv2.putText(frame,str(box_count_num),(int(i[0])-40,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv2.LINE_AA) 
                                cv2.line(frame,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)

                
               
                cv2.putText(frame,"Count => "  + str(len(detected_object_name_list)),(60,680),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) 
                count = str(len(detected_object_name_list))
                
                box_count_num = 0
                detected_object_name_list = []
                frame = frame
            else:
                cv2.putText(frame,"Count => "  + count ,(60,680),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) 

            img_show.image(frame)
    

if choice == "Real Time":
    
    camera_input_data = st.camera_input("Take a pic")

    if camera_input_data is not None:
        
        model = YOLO("weights/"+ model_choice +".pt")   
        classes_list = model.names
        detected_object_name_list = []
        box_count_num = 0

        image = Image.open(camera_input_data)
        image.save("input_data_image.png")
        frame = cv2.imread("input_data_image.png")
                
        # model prediction
        results = model.predict(source=frame,iou=0.7,conf=conf)
        
        get_array = results[0].boxes.numpy().boxes.tolist()

        # function to sort array 
        get_array.sort(key=sort_array_func)
        
        #------------------------------------------- processing the frame some condition ----------------------------------------#
        if len(get_array) == 0:
            pass
        else:
            for ind,i in enumerate(get_array):

                    # overlap condition 
                    if len(get_array) != ind+1 :
                        
                        # print(ind+1," ==> ",int((int(get_array[ind+1][1])-i[1])))
                        if int((int(get_array[ind+1][1])-i[1])) > 12:
                            
                        # if 1==1:
                            last = classes_list[round(i[-1])]
                            detected_object_name_list.append(last)
                            box_count_num += 1
                            cv2.rectangle(frame,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(0, 255, 0), 2)
                            cv2.putText(frame,str(box_count_num),(int(i[0])-70,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2,cv2.LINE_AA) 
                            cv2.line(frame,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)
                            
                    else:
                        
                        last = classes_list[round(i[-1])]
                        detected_object_name_list.append(last)
                        box_count_num += 1
                        
                        cv2.rectangle(frame,(int(i[0]),int(i[1])), (int(i[2]), int(i[3])),(0, 255, 0), 2)
                        cv2.putText(frame,str(box_count_num),(int(i[0])-40,int(i[1])+10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv2.LINE_AA) 
                        cv2.line(frame,pt1=(int(i[0]),int(i[1])+10),pt2=(int(i[0])-40,int(i[1])+5),color=(0,0,255),thickness=2)

        cv2.putText(frame,"Count => "  + str(len(detected_object_name_list)),(10,680),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) 

        box_count_num = 0
        detected_object_name_list = []

        st.image(frame)