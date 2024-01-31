from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import numpy as np
import os
import sys
from pathlib import Path
from detect_wrapper.Detectoruav import DroneDetection
from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker
import settings
import cv2
import time
import logging

from typing import Annotated

sys.path.append(os.path.join(os.path.dirname(__file__),'detect_wrapper'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\dronetracker'))
sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\drtracker'))

log = logging.getLogger('uvicorn')

models ={}

# load detection, tracking models
@asynccontextmanager
async def lifespan(app: FastAPI):
    sys.path.append(os.path.join(os.path.dirname(__file__),'detect_wrapper'))
    sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\dronetracker'))
    sys.path.append(os.path.join(os.path.dirname(__file__),'tracking_wrapper\\drtracker'))

    IRweights_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'detect_wrapper/weights/best.pt')
    assert Path(IRweights_path).exists(), "Path doestn exists:" + IRweights_path
    log.info(f"Loading detection model on {IRweights_path=}")
    drone_det = DroneDetection(IRweights_path=IRweights_path, RGBweights_path=IRweights_path)
    drone_tracker = Tracker()
    
    models['drone_det'] = drone_det
    models['drone_tracker'] = drone_tracker
    log.info(f'Init ... {list(models.keys())=}')
    
    yield
    
    models.clear()    

app = FastAPI(lifespan=lifespan)

security = HTTPBasic()

# authentification
def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"elenavatkina"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"mathematica"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/users/me")
def read_current_user(username: Annotated[str, Depends(get_current_username)]):
    return {"username": username}

@app.get("/")
def read_root():
    return {"Hello": "World"}
 
 # process video, get result   
@app.get("/tracking/{video_file_name}")
async def track(video_file_name: str):
    try:
        vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(video_file_name)))
            
        det_time=[]
        time_record=[]
        first_track=True
        magnification = 2
        interval=50
        
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:                   
                oframe = image.copy()
                t1=time.time()
                # log.info(f'Detecting... {list(models.keys())=}')
                init_box = models['drone_det'].forward_IR(image)
                # log.info('Detecting...2')  
                t2=time.time()
                det_time.append(t2-t1)
                    
                if init_box is not None:
                    if first_track:
                        models['drone_tracker'].init_track(init_box,image)
                    else:
                        models['drone_tracker'].change_state(init_box)
                    bbx = [int(x) for x in init_box]

                    visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
                    bbx=[i*magnification for i in bbx]
                    cv2.rectangle(oframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)        
                        
                num=0
                while(num<interval):
                    num=num+1
                    ret, frame = vid_cap.read()      

                    if ret:
                        oframe = frame.copy()
                        visuframe = cv2.resize(oframe, (oframe.shape[1]*magnification, oframe.shape[0]*magnification), cv2.INTER_LINEAR)
                        t1=time.time()
                        outputs = models['drone_tracker'].on_track(frame) 
                        t2=time.time()
                        time_record.append(t2-t1)
                        bbx=[i*magnification for i in outputs]    
                        cv2.rectangle(visuframe,(bbx[0],bbx[1]), (bbx[0]+bbx[2],bbx[1]+bbx[3]), (0,255,0), 2)  
                    
            else:
                vid_cap.release()
                    
                track_avrg_time = round(np.array(time_record).mean(), 6)
                detect_avrg_time = round(np.array(det_time).mean(), 6)
                               
                break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

    return {
        "status":200,
        "track_time": track_avrg_time, 
        "detect_time": detect_avrg_time
    }
# run in terminal
# uvicorn main:app --reload    use without reload!
