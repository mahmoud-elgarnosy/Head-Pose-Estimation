import joblib
import cv2
from math import cos, sin
from LandMarks import landMarks
from NormalizePoints import noramlize_points



class drawAxis():
    def __init__(self) -> None:
        #Loading model and classes
        self.nm = noramlize_points()
        self.loaded_model = joblib.load('./finalized_model_v6.sav')
        self.lm = landMarks()


    def yaw_pitch_roll_prediction(self,land_marks):
        self.loaded_model.predict(land_marks)

    def draw_axis(self,img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        yaw = - yaw 
        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2 
        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy 
        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy 
        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy 
        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),3) 
        return img

    def show_img(self,frame):
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        land_marks,tdx,tdy = self.lm.face_land_marks(cv2_frame)
           
        #check if there is any land_marks not to be a None
        if land_marks.any().any():

            #first normalize points
            land_marks = self.nm.transform(land_marks)
            
            #draw axis on predicted (pitch, yaw ,roll)
            pitch, yaw ,roll = self.loaded_model.predict(land_marks)[0]
            frame = self.draw_axis(frame, yaw, pitch, roll,tdx,tdy)
        return frame

