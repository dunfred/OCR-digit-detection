import cv2
import pytesseract
import numpy as np
OUTPUT_SIZE_WIDTH = 2560
OUTPUT_SIZE_HEIGHT = 1920

class dfb:     #Draw focus bar class
    def __init__(self, img):            
        self.x = int(img.shape[0] * (35/100))  #Top
        self.x1 = int(img.shape[0] * (38/100))

        self.y = int(img.shape[1] * (65/100)) #Right
        self.y1 = int(img.shape[1] * (63/100))

        self.w = int(img.shape[0] * (35/100)) #left
        self.w1 = int(img.shape[0] * (38/100))


        self.h = int(img.shape[1] * (35/100)) #Down
        self.h1 = int(img.shape[1] * (33/100))
        
        self.rect = cv2.rectangle(img,(self.x,self.w),(self.y,self.h),(255,255,0),3)
        self.rect2 = cv2.rectangle(img,(self.x1,self.w1),(self.y1,self.h1),(155,155,0),2)
        

cap = cv2.VideoCapture(0) #Capture video from camera

while True:
    ret, frame = cap.read()
    #Resize the duplicated frame 1920 x 2560
    resized = cv2.resize(frame,(OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT), interpolation = cv2.INTER_LINEAR)
    #Draw the recharge card number focus box for the frame
    frame_f_rect =  dfb(frame).rect2
    #Show image
    cv2.imshow("Capture The Recharge Card", frame_f_rect)
        
    img = frame_f_rect
    digit = img[dfb(img).x1 : dfb(img).h1, dfb(img).w1: dfb(img).y1]

    #Extract text and store in text_data variable
    text_data = pytesseract.image_to_string(digit, lang='eng')    

    recharge_digits = ''
    #Check if all returned values are numbers
    if len(text_data) >= 14:
        for i in text_data:
            try:
                int(i)/1          #Checking if i is a number
            except Exception as e:
                pass
            else:                
                recharge_digits += i
        #If length of returned values is equall to 14, print them to console
        if len(recharge_digits) == 14:
            print("Your Recharge card is: ", recharge_digits)                   

    #If user presses "Q" key, stop capture and close all windows
    if cv2.waitKey(1) & 0xFF == ord("q"):        
        break

cap.release()
cv2.destroyAllWindows()


