import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

class cropFaces():
    def __init__(self) -> None:
       self.mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, # model selection
        min_detection_confidence=0.2# confidence threshold
)

    def crop_faces(self,dframe):

        #extract bounding box around face
        image_rows, image_cols, _ = dframe.shape
        image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(image_input)
        if results.detections:
            
            #extract first face
            detection=results.detections[0]
            location = detection.location_data

            relative_bounding_box = location.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                image_rows)
            rect_end_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                image_rows)


            #check if there is a bounidng box 
            if(rect_end_point !=None and rect_start_point != None  ):
                xleft,ytop=rect_start_point
                xright,ybot=rect_end_point

                crop_img = image_input[ytop: ybot, xleft: xright]
                return crop_img

        return image_input