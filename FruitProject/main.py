import cv2
import numpy as np


# this function is similar the Select Roi commum of opencv, but it accepts a image that represents whole, and positions of bounding boxes of a subimage, case needs to get of a part of the image   
def select_roi(window_name, whole_image, sub_image_bounds=None):
    """
    Displays a window with the given name and allows the user to select a rectangular region of interest (ROI) from the image.
    If the sub_image_bounds argument is provided, the ROI selection is restricted to the sub-image defined by the bounding box.
    Returns a list [roi_x, roi_y, roi_width, roi_height] representing the (x, y) coordinates of the top-left corner of the ROI and its width and height.
    """
    if window_name is None or len(window_name) == 0:
        raise ValueError("Invalid window_name argument")
    if not isinstance(whole_image, np.ndarray):
        raise TypeError("Invalid whole_image argument")

    if sub_image_bounds is not None:
        selected_sub_image = whole_image[sub_image_bounds[1]:sub_image_bounds[1] + sub_image_bounds[3], sub_image_bounds[0]:sub_image_bounds[0] + sub_image_bounds[2]]
    else:
        selected_sub_image = whole_image

    roi_positions = cv2.selectROI(window_name, selected_sub_image)
    cv2.destroyWindow(window_name)

    roi_x = sub_image_bounds[0] + roi_positions[0] if sub_image_bounds is not None else roi_positions[0]
    roi_y = sub_image_bounds[1] + roi_positions[1] if sub_image_bounds is not None else roi_positions[1]
    roi_width = roi_positions[2]
    roi_height = roi_positions[3]

    return [roi_x, roi_y, roi_width, roi_height]

class Process:
    SIDE_REAL_SQUARE = 2 #(m)
    FILTER_BAD_CONTOURS = 2.3 # (cm)
    RUNNING = True
    
    # make a method to check if the object is a square, because the reference has this shape    
    def is_square(self, contour):
        width, height = cv2.boundingRect(contour)[2:]
        if height < 100: return
        aspect_ratio = width / height
        return 0.9 < aspect_ratio < 1.1   
    
    # convert the pixels dimensions to real dimensions
    def get_real_dimension(self, pixels_dimensions_object, side_square_reference):
        pixels_dimensions_object_arr = np.array(pixels_dimensions_object)          
        return (pixels_dimensions_object_arr * self.SIDE_REAL_SQUARE) / side_square_reference
    
    # check if a contour is a eligible fruit   
    def is_possible_fruit(self, contour, side_square_reference):
        real_fruit_dimensions = self.get_real_dimension(contour, side_square_reference) 
        return np.all((4 < real_fruit_dimensions, real_fruit_dimensions < 8))
    
    def mouse_callback(self, event, x, y, flags, param):
        # this event will works if the left button of the mouse is clicked and if the click was inside the fruit contour
        if event == cv2.EVENT_LBUTTONDOWN:
            # make select roi in image of contour of the fruit to get the positions of the deformations:
            if cv2.pointPolygonTest(self.fruit_contour, (x, y), False) == 1:
                print("clicked inside the fruit contour")
                fruit_bounding_box = cv2.boundingRect(self.fruit_contour) 
                current_frame = param['current_frame']
                self.deformation_bounding = select_roi("Select the defomation bounding", current_frame, fruit_bounding_box)
    def __init__ (self):
        # self.capture = None
        self.cap = cv2.VideoCapture(0) # por enquanto
        cv2.namedWindow("frame")
        self.object_reference_contour = None
        self.fruit_contour = None
        self.deformation_bounding = None 
        self.clicked_inside_fruit = False 

    def start(self):
        while self.RUNNING and self.cap.isOpened():
            if not self.cap.isOpened():
                print("Capture is not working")
                break
            frame = self.cap.read()[1]
            # get bin using otsu 
            gray_image = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
            bin = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # apply the edge detection
            contours = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            # make the dection of normalization object:
            
            # getting all the possible squares:
            square_contours  = [contour for contour in contours if self.is_square(contour)]
            # setting the biggest square as the object reference:  
            if square_contours:
                self.object_reference_contour = max(square_contours, key= cv2.contourArea) 
                obj_reference_bounding_box = cv2.boundingRect(self.object_reference_contour) 
                cv2.rectangle(frame, obj_reference_bounding_box, (0, 255, 0), 2)
                print("Object Reference Detected")
                # this part needs to be tested, can be removing good contours !?!?!?!? 
                # contours = contours[np.array_equal(contours, square_contours)]
                contours = contours[not np.array_equal(contours, self.object_reference_contour)] 
                               
                # Now that has the reference, can gets possibles fruit object           
                possible_fruit_contours = [contour for contour in contours if self.is_possible_fruit(contour, obj_reference_bounding_box[1])]
                # setting the biggest possible fruit as the fruit object:
                if possible_fruit_contours:
                    self.fruit_contour = max(possible_fruit_contours, key= cv2.contourArea)
                    fruit_bounding_box = cv2.boundingRect(self.fruit_contour)
                    frame = cv2.rectangle(frame, fruit_bounding_box, (255, 0, 0), 2)
                    # get the real dimensions of the fruit
                    real_dimensions_fruit = self.get_real_dimension(fruit_bounding_box[2:], obj_reference_bounding_box[3])
                    print("Fruit Detected\nWidth: %d\nHeight: %d\n" % (real_dimensions_fruit[0], real_dimensions_fruit[1]))
                    
                    cv2.setMouseCallback("frame", self.mouse_callback, param = {'current_frame': frame})
                    if self.deformation_bounding:
                        cv2.rectangle(frame, self.deformation_bounding, (0, 0, 255), 2)
                        print("User Select The Deformation")
                        # apply the calculate of firmness(Fz), where Fz = Mass(M) / Area (A) * 9.8, and M = 0.264, and A = (0.784 * c1 * c2)    
                        # getting the reals dimensions of length c1 and c2 of deformation bound
                        c1, c2 = self.get_real_dimension(self.deformation_bounding[2:], obj_reference_bounding_box[3])
                        print("Longer Length: %d\nShorter Length: %d" % (c1, c2))
                        # calculate the area
                        area = (0.784 * c1 * c2)
                        # calculate the firmness
                        firmness = (0.264 / area) * 9.8
                        print("Firmness: %d" % firmness)
                else:
                    # case has no fruit, bounding of deformation needs to be reseted
                    self.deformation_bounding = None 
                                          
            cv2.imshow("frame", frame)    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break              