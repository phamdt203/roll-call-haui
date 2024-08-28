import os
import cv2
import uvicorn
import numpy as np
from config import cfg
from pipeline import Pipeline
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title = "Face recognition")

pline = Pipeline()

@app.post("/check/")
async def check_face(image: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    userIDs,_,_ = pline.check(img, cfg["thresh_angle"])
    
    names = {idx : {'name' : userID.split('_')[0], 'id' : userID.split('_')[1]} if userID else {idx : "strange"} for idx, userID in enumerate(userIDs)}
    if len(names):
        names["message"] = "CHECK FACE SUCESSFULLY"
    return names

@app.post('/add-new/')
async def add_face(name: str, student_id: str, image: UploadFile = File()):
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"{cfg['data_path']}/{name}_{student_id}.jpg", img)

@app.get('/reload/')
async def reload() :
    pline.reload(pline.retinaface, pline.arcface)

if __name__ == '__main__':
    uvicorn.run(app, host = "0.0.0.0", port = 8000)