# import the opencv library
import cv2
from DrawAxis import drawAxis
  
DA = drawAxis()

vid = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output4.mp4', fourcc, 20.0, (640,480))
c = 1
while(True):
      
    # Capture the video frame
    # by frame
    
    ret, frame = vid.read()

    #draw axis on a frame
    frame = DA.show_img(frame)
    
        
    #
    cv2.imshow('frame', frame)
    out.write(frame)

    
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
out.release()
# Destroy all the windows
cv2.destroyAllWindows()