# from tkinter.messagebox import NO
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from CropFaces import cropFaces

class landMarks():
    def __init__(self):
        self.crop_face = cropFaces()
        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         min_detection_confidence=.3)
        mp_face_detection = mp.solutions.face_detection
        self.face_detector =  mp_face_detection.FaceDetection( min_detection_confidence = 0.6)

    def face_land_marks(self,img):
        # intialize variable
        landmarks = pd.DataFrame(np.array([]).reshape(1,-1))
        tdx,tdy = None,None

        #crop faces and resize image
        img_2 = self.crop_face.crop_faces(img)
        img_2 = cv2.resize(img_2, (300,300), interpolation = cv2.INTER_AREA)

        #exract land marks from face
        face_mesh_results = self.face_mesh_images.process(img_2[:,:,::-1])
        if face_mesh_results.multi_face_landmarks:
            results = self.face_detector.process(img)

            #check if there is a face
            if(results.detections):
                landmark = results.detections[0].location_data.relative_keypoints

                #draw axis on nose location 
                tdx,tdy = (int(landmark[2].x * img.shape[1]), int(landmark[2].y * img.shape[0]))
                
                #createing dataframe from land marks
                landmarks_x = [face_mesh_results.multi_face_landmarks[0].landmark[i].x for i in range(468)]
                landmarks_y = [face_mesh_results.multi_face_landmarks[0].landmark[i].y for i in range(468)]
                landmarks = landmarks_x + landmarks_y
                landmarks = pd.DataFrame(np.array(landmarks).reshape(1,-1))
                # return landmarks
        return landmarks,tdx,tdy
