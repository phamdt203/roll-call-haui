import os
import cv2
import numpy as np
from database import DataBase
from typing import Optional
from model.face.arcface import ArcFace
from model.face.landmark import Landmark
from model.face.retinaface import RetinaFace

class Pipeline:
    def __init__(self) -> None:
        self.arcface = ArcFace('arcface_s')
        self.retinaface = RetinaFace('retina_s')
        self.landmark = Landmark('2d')
        self.database = DataBase('data')
        self.database.reload(self.retinaface, self.arcface)

    def check(self, img : np.ndarray, thresh_angle : Optional[int] = None) -> tuple:
        boxes, kpoints = self.retinaface.detect(img, max_num=100)

        idxs = []
        for idx, box in enumerate(boxes):
            land = self.landmark.get(img, box)
            angle = self.landmark.get_face_angle(img, land, False)[1]
        
            if abs(angle) < thresh_angle:
                idxs.append(idx)
        
        userIDs = []
        for kpoint in kpoints[idxs]:
            em = self.arcface.get(img, kpoint)
            userID = self.database.find(em)
            
            if userID is not None:
                userIDs.append(userID)
            else:
                userIDs.append(None)

        return userIDs, boxes[idxs], kpoints[idxs]

    def reload(self) -> None:
        self.database.reload(self.retinaface, self.arcface)

    def add_new_face(self, userID : str, name: str, img : np.ndarray) -> None:
        _, keypoints = self.retinaface.detect(img)

        if keypoints is not None:
            emb = self.arcface.get(img, keypoints[0])
            
            self.database.add_new_face(userID, emb)

            cv2.imwrite("data/{name}_{userID}.jpg")       