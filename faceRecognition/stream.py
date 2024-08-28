import cv2
import time
from pipeline import Pipeline 

pline = Pipeline()    
camera_ip = "172.16.17.187"  
username = "admin"
password = "DHCN@246"  
RTSP_URL = f"rtsp://{username}:{password}@{camera_ip}/stream/1"
cam = cv2.VideoCapture(0)  

resized_width = 640  
resized_height = 480  

fps_counter = 0  
start_time = time.time()  
fps_start_time = time.time()  

while True:  
    ret, frame = cam.read()  

    if not ret:  
        print("End of video stream.")  
        break  

    frame = cv2.resize(frame, (resized_width, resized_height))  

    picture = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    userIDs, _, _ = pline.check(picture, 15)
    if userIDs:  
        cv2.putText(frame, userIDs[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  
    fps_counter += 1  
    current_time = time.time()
    if current_time - fps_start_time >= 1.0:  
        fps = fps_counter / (current_time - fps_start_time)  
        print(f"FPS: {fps:.2f}")  
        fps_counter = 0  
        fps_start_time = current_time  

    cv2.imshow('Video Stream', frame)  
 
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

cam.release()  
cv2.destroyAllWindows()