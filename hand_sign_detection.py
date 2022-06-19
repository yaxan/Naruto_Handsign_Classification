import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)
box_width = 300

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if True:
        pass

    # Bounding box for hands to be placed in
    height = frame.shape[0]
    width = frame.shape[1]

    left = int((width-box_width)/2)
    top = int(height/2)
    right = left + box_width
    bottom = height

    #process_this_frame = not process_this_frame
    cropped_frame = frame[top:bottom, left:right, :]

    #TODO run ML on cropped_frame
    #TODO display text

    # test the cropped image box
    # cv2.imwrite(os.path.join(os.curdir, "testCroppedImage.jpg") , cropped_frame)

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # Display the resulting image
    cv2.imshow('Video', frame)
    



    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
    
    
