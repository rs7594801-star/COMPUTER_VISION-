import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 
import math
MODEL_PATH  = 'face_landmarker.task'
EAR_THRESHOLD  = 0.20
CONSEC_FRAMES = 2

LEFT_EYE_IDX = [362 , 385 , 387 , 263 , 373 , 380]
RIGHT_EYE_IDX = [33 , 160 , 158 , 133 ,153 ,144]


blink_c  = 0 
frame_counter  = 0 
def draw_eye_rect(frame, pts, eye_indices, color, w, h):
    coords = []
    for idx in eye_indices:
        point = pts[idx]
        coords.append((int(point.x * w), int(point.y * h)))
    
    coords = np.array(coords)
    x_min, y_min = coords.min(axis=0) 
    x_max, y_max = coords.max(axis=0) 

    
    padding = 5
    cv2.rectangle(frame, 
                  (x_min - padding, y_min - padding), 
                  (x_max + padding, y_max + padding), 
                  color, 1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

capture = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(options) as landmarker :
    while capture is not None :
        success , frame = capture.read()
        if not success :
            break
        frame = cv2.flip(frame , 1 )


        h ,w , _ = frame.shape
        # frame = cv2.flip(frame , 1 )
        rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB , data = rgb)
        timestamp = int(time.time()* 1000)


        result = landmarker.detect_for_video(mp_image, timestamp)
        # detector = vision.FaceLandmarker.create_from_options(options)
        # detection_result = detector.detect(mp_image)

        # results = landmarker.detect_for_video(mp_image , timestamp)

        if result.face_landmarks :
            
            pts = result.face_landmarks[0]
            # if detection_result.face_landmarks:
            #         face_landmarks = detection_result.face_landmarks[0]

                

                    # face_landmarks = detection_result.face_landmarks[0]
            pts = result.face_landmarks[0]
            p_thumb = pts[10]
            p_index = pts[152]
            side_cheekL = pts[234]
            side_cheekR = pts[454]
                
            cx_t, cy_t = int(p_thumb.x * w), int(p_thumb.y * h)
            cx_i, cy_i = int(p_index.x * w), int(p_index.y * h)
            ckx,cky  = int(side_cheekL.x * w), int(side_cheekL.y * h)
            ck1, ck2 = int(side_cheekR.x * w), int(side_cheekR.y * h)
            angle_rad = math.atan2(cy_i - cy_t, cx_i - cx_t)
            angle_deg = math.degrees(angle_rad)
            cv2.line(frame, (cx_t, cy_t), (cx_i, cy_i), (255, 0, 255), 3)
            cv2.line(frame, (ckx , cky), (ck1, ck2), (255, 0, 255), 3)
            cv2.circle(frame, (cx_t, cy_t), 3, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (cx_i, cy_i), 3, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (ckx, cky), 3, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (ck1, ck2), 3, (0, 255, 0), cv2.FILLED)


            dial_center = (w - 100, 100)
            dial_radius = 50
                    
                    
            # cv2.circle(frame, dial_center, dial_radius, (0,0,255), 2)
            needle_x = int(dial_center[0] + dial_radius * math.cos(angle_rad))
            needle_y = int(dial_center[1] + dial_radius * math.sin(angle_rad))
            cv2.line(frame, dial_center, (needle_x, needle_y), (0, 255, 255), 4)
            if angle_deg <= 92 or angle_deg >= 89 :
                cv2.line(frame, dial_center, (needle_x, needle_y), (0, 0, 255), 4)
                
            cv2.putText(frame, f"Rotation: {int(angle_deg)} deg", (dial_center[0]-80, dial_center[1]+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if angle_deg == 90 :
                cv2.putText(frame, f"Rotation: {int(angle_deg)} deg", (dial_center[0]-80, dial_center[1]+80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            
            l =  [np.array([pts[i].x * w, pts[i].y * h]) for i in LEFT_EYE_IDX]
            l_v1 = np.linalg.norm(l[1] - l[5])
            l_v2 = np.linalg.norm(l[2] - l[4])
            l_h  = np.linalg.norm(l[0] - l[3])
            left_ear = (l_v1 + l_v2) / (2.0 * l_h)


            r  = [np.array([pts[i].x * w , pts[i].y * h]) for i in RIGHT_EYE_IDX]
            r_v1 = np.linalg.norm(r[1] - r[5])
            r_v2 = np.linalg.norm(r[2] - r[4])
            r_h  = np.linalg.norm(r[0] - r[3])
            right_ear = (r_v1 + r_v2) / (2.0 * r_h)


            left_ear  = (l_v1 + l_v2) / (2.0 * l_h)
            right_ear= (r_v1 + r_v2) / (2.0 *r_h)

            avg_ear = (left_ear + right_ear)/ 2 



            red = (0,0,255)
            green =(144,238,144)
            l_color = red if left_ear < EAR_THRESHOLD else green 
            draw_eye_rect(frame , pts ,LEFT_EYE_IDX , l_color , w , h)
            if right_ear < EAR_THRESHOLD and left_ear < EAR_THRESHOLD:
                cv2.putText(frame , "BOTH CLOSED ",(30,200) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2 )
            if left_ear < EAR_THRESHOLD:
                cv2.putText(frame , "RIGHT EYE" ,(w - 180,100), cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2) 
                # cv2.putText(frame , "ONE EYE CLOSED " ,(w - 180,160), cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2) 


            r_color = red if right_ear < EAR_THRESHOLD else green 
            draw_eye_rect(frame , pts , RIGHT_EYE_IDX , r_color , w ,h)
            
            # if right_ear < EAR_THRESHOLD and left_ear < EAR_THRESHOLD:
            #     cv2.putText(frame , "BOTH CLOSED ",(30,200) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2 )

            if  right_ear < EAR_THRESHOLD:
                cv2.putText(frame , "LEFT EYE  ",(30,100) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2 )
                # cv2.putText(frame , "ONE EYE CLOSED ",(30,160) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2 )
         
            if right_ear < EAR_THRESHOLD or left_ear < EAR_THRESHOLD:
                cv2.putText(frame , "ONE EYE CLOSED ",(30,160) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2 )
            
                              


                   

                    # cv2.putText(frame , "BOTH EYES ARE CLOSED  ",(30,190) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 2 )
        
            if avg_ear < EAR_THRESHOLD :
                frame_counter += 1
                # cv2.putText(frame , "BOTH EYES CLOSED",(200 , 200) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 1 )
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_c+=1
                    # cv2.putText(frame , "BOTH EYES CLOSED",(200 , 200) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 1 )

                frame_counter = 0 
                # cv2.putText(frame , "BOTH EYES CLOSED",(200 , 200) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 , red , 1 )

    
        cv2.putText(frame , f"BLINKD : {blink_c}" , (30,50) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,255), 2 )

        cv2.imshow("LEFT RIGHT DETECTOR AND BLINK COUNTER",frame )
        # cv2.putText(frame , f"BLINKS : {blink_c}" , (30,50) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255), 2 )
           
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()










            






