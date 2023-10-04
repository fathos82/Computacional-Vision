
import cv2
import numpy as np


class Process:
    SIDE_REAL_SQUARE = 2 #(cm)
    FILTER_BAD_CONTOURS = 2.3 # (cm)
    
    
    # make a method to check if the object is a square, because the reference has this shape    
    def is_square(self, contour):
        
        width, height = cv2.boundingRect(contour)[2:]
        if height < 100: return
        aspect_ratio = width / height
        return 0.9 < aspect_ratio < 1.1   
    
    # convert the pixels dimensions to real dimensions
    def get_real_dimension(self, object_contour, side_square_reference):
        pixels_dimensions_object = np.array(cv2.boundingRect(object_contour)[2:])
        return (pixels_dimensions_object * self.SIDE_REAL_SQUARE) / side_square_reference
    
    # check if a contour is a eligible fruit   
    def is_possible_fruit(self, fruit_contour, side_square_reference):
        real_fruit_dimensions = self.get_real_dimension(fruit_contour, side_square_reference) 
        return real_fruit_dimensions > 4 and real_fruit_dimensions < 8
    
    def mouse_callback(self, event, x, y, flags, contour):
        if event == cv2.EVENT_LBUTTONDOWN:
            #q: why servers the "pointPolygonTest"?
            #
            if cv2.pointPolygonTest(contour, (x, y), False) == 1:
                print("Hello")
        
    def __init__ (self):
        # self.capture = None
        self.cap = cv2.VideoCapture(0) # por enquanto
        
    def start(self):
        while True:
            if not self.cap.isOpened():
                print("Capture is not working")
                break
            
            frame = self.cap.read()[1]
            # get bin using otsu 
            gray_image = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
            bin = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # apply the edge detection
            contours = np.array(cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
            
            # make the dection of normalization object:
            
            # getting all the possible squares:
            square_contours  = [contour for contour in contours if self.is_square(contour)]
            # setting the biggest square as the reference:  
            if square_contours:
                object_reference = max(square_contours, key= cv2.contourArea) 
                contours = contours[contours != square_contours]
                side_square_reference = cv2.boundingRect(object_reference)[2:]
                
                # Now that has the reference, can gets the fruit object:                
                possible_fruit_contours = [contour for contour in contours if self.is_possible_fruit(contour, object_reference)]
                
                # setting the biggest possible fruit as the fruit object:
                if possible_fruit_contours:
                    fruit_contour = max(possible_fruit_contours, key= cv2.contourArea)
                    real_dimensions_fruit = self.get_real_dimension(fruit_contour, object_reference)
                    frame = cv2.rectangle(frame, cv2.boundingRect(fruit_contour), (255, 0, 0), 2)
                    # i would like a method that if the user clicks in bounding box of fruit, prints "Hello":
                    l =  lambda event, x, y, flags, param: print(x)
                    cv2.setMouseCallback("frame", self.mouse_callback, l)
                    
                    
                    
                                                          
        
                

            cv2.imshow("frame", frame)    
                
                        
                
            
            
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break              
    

    
p = Process() 
p.start()

    

