import cv2
import numpy as np
import streamlit as st
from config import cfg
from pipeline import Pipeline

pline = Pipeline()

def main():
    picture = st.camera_input("Take a stream")
    picture = cv2.imdecode(np.frombuffer(picture.read(), np.uint8), -1)
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    thresh_angle = 15
    userIDs,_,_ = pline.check(picture, thresh_angle)
    names = [userID.split('_') if userID else "strange" for userID in userIDs]
    st.write(names)
    
if __name__ == "__main__":
    main()