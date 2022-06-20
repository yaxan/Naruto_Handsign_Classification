import cv2 as cv
import os

from cv2 import waitKey

#Sets to default camera
camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Camera not opened, terminating..")
    exit()

#Add/change/remove labels
Labels = ["bird", "boar", "dog", "dragon", "Hare", "horse", "Monkey", "ox", "Ram", "Rat", "Snake", "Tiger"]

for label in Labels:
    if not os.path.exists(label):
        os.mkdir(label)

for folder in Labels:

    count = 0

    print("Press 'x' for "+folder)
    userinput = input()
    if userinput != 'x':
        print("Wrong Input, terminating...")
        exit()

    #Delay before taking pictures
    waitKey(2000)
    
    # Change number of pictures per label
    while count<500:

        status, frame = camera.read()

        if not status:
            print("Frame is not been captured, terminating...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("Video Window",gray)
        gray = cv.resize(gray, (600,400))
        cv.imwrite('D:/code/test/'+folder+'/img'+str(count)+'.png',gray)
        count=count+1
        waitKey(100)
        if cv.waitKey(1) == ord('q'):
            break

camera.release()
cv.destroyAllWindows()