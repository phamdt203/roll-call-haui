import os
import cv2
import numpy as np

class DataBase:
    def __init__(self, path_data: str) -> None:
        self.path_data = self.__get_path(path_data)

    def __get_path(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def reload(self, face_model, embed_model) -> None:
        database_emb = {
            'userID': [],
            'embs': []
        }

        img_data_list = os.listdir(self.path_data)
        for path in img_data_list:
            img_path = os.path.join(self.path_data, path)
            img = cv2.imread(img_path)
            _, keypoint_data = face_model.detect(img, max_num=1)
            emb_data = embed_model.get(img, keypoint_data[0])
            
            database_emb['embs'].append(emb_data)
            database_emb['userID'].append(path.split('.')[-2])
        print('Extract feature on databse done!')

        database_emb['embs'] = np.array(database_emb['embs'])
        database_emb['userID'] = np.array(database_emb['userID'])

        self.embs = database_emb

    def add_new_face(self, userID: str, emb: np.ndarray):
        if userID in self.embs['userID']:
            idx = np.where(self.embs['userID'] == userID)[0][0]
            self.embs['embs'][idx] = emb
        else:
            if self.embs['embs'].shape[0] == 0:
                self.embs['embs'] = np.expand_dims(emb, axis=0)
            else:
                self.embs['embs'] = np.append(self.embs['embs'], [emb], axis=0)
            self.embs['userID'] = np.append(self.embs['userID'], [userID], axis=0)
    
    def find(self, emb: np.ndarray, thresh=0.3):
        if self.embs['embs'].shape[0] == 0:
            return None
        dis = np.dot(self.embs['embs'], emb) / (np.linalg.norm(self.embs['embs'], axis=1) * np.linalg.norm(emb))
        if np.max(dis) > thresh:
            idx = np.argmax(dis)
            return self.embs['userID'][idx]
        else:
            return None
        