# 
import cv2
import streamlit as st
import settings
import time
import os
import numpy as np
import logging

from pathlib import Path
# model for detection
from detect_wrapper.Detectoruav import DroneDetection
# model for tracking
from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

# choose and process video
def play_stored_video():
    log = logging.getLogger('play_stored_video')
    
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Track Object'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            
            det_time=[]
            time_record=[]
            first_track=True
            magnification = 2
            interval=50
            
            IRweights_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'detect_wrapper/weights/best.pt')
            log.info(f"{IRweights_path=}")
            assert Path(IRweights_path).exists(), "Path doestn exists:" + IRweights_path
            drone_det = DroneDetection(IRweights_path=IRweights_path, RGBweights_path=IRweights_path)
            drone_tracker = Tracker()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    
                    oframe = image.copy()
                    t1=time.time()
        
                    init_box=drone_det.forward_IR(image)
                    
                    t2=time.time()
                    det_time.append(t2-t1)
                    
                    if init_box is not None:
                        if first_track:
                            drone_tracker.init_track(init_box,image)
                        else:
                            drone_tracker.change_state(init_box)
                        bbx = [int(x) for x in init_box]
                        # log.info('detector {bbx=}')
                        # visualize result box
                        visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
                        bbx=[i*magnification for i in bbx]
                        cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)
                        
                        st_frame.image(visuframe,
                                    caption='Tracked object',
                                    channels="BGR",
                                    use_column_width=True
                                    )
                        
                    num=0
                    while(num<interval):
                        num=num+1
                        ret, frame = vid_cap.read()      

                        if ret:
                            oframe = frame.copy()
                            visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
                            t1=time.time()
                            outputs = drone_tracker.on_track(frame) 
                            t2=time.time()
                            time_record.append(t2-t1)
                            bbx=[i*magnification for i in outputs]
                            # log.info('tracker {bbx=}')

                            cv2.rectangle(visuframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)  
                            
                            st_frame.image(visuframe,
                                        caption='Tracked object',
                                        channels="BGR",
                                        use_column_width=True
                                        )
                    
                else:
                    vid_cap.release()
                    
                    track_avrg_time = np.array(time_record).mean()
                    detect_avrg_time = np.array(det_time).mean()
                    
                    # return track, detect time
                    st.write('track average time:', round(track_avrg_time, 6))
                    st.write('detect average time:', round(detect_avrg_time, 6))
                               
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    
        
