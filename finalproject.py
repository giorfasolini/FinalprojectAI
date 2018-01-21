import cv2 #Import openCV library for data training, and face detection
import os #Import os library for reading directories and image paths
import numpy as np #Import numpy library for the use of dimentional arrays and matrices

# Define subject name in test-case folder
subjects = ["", "Jennifer Lawrence", "Bill Gates", "Bruce Willis", "David Beckam", "Hugh Jackman"]


#function to detect face using OpenCV
def detect_face(img):
    #Converting to gray by removing its hue and saturtion while retaining the luminance because gray color
    #is better for object detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Load xml file for dataset LBP (Local Binary Pattern) as defined in xml file
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #Detect all images in one image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# Read all training images
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    for dir_name in dirs:
        
        #All training data start with "s" in a folder
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #Get a folder label from training data
        label = int(dir_name.replace("s", ""))
        
        #Specify the training data directory
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #Get the all image names on the folder
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing images")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))


#OpenCV library for face recognition using LBPH (Local Binary Pattern Histogram)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# There are optional face recognizer as EigenFaces or FisherFaces.
# However if using EigenFaces & FisherFaces, all samples must on equal size

#Train face recognizer
face_recognizer.train(faces, np.array(labels))


#Drawing rectangle around image
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #Defining parameter (img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)
    
#Putting text on above image
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    #Defining parameter (img, text, startPoint, font, fontSize, rgbColor, lineWidth)


#Predict similar faces in test-data based of data training
def predict(test_img):
    img = test_img.copy() 
    
    face, rect = detect_face(img)    
    label, confidence = face_recognizer.predict(face)
    
    label_text = subjects[label]
    
    
    draw_rectangle(img, rect)  
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Predicting images...")

#Load images from directories
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")
test_img4 = cv2.imread("test-data/test4.jpg")
test_img5 = cv2.imread("test-data/test5.jpg")

#Predict images
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
print("Prediction complete")

#Show images on 400x500 pixel
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.imshow(subjects[4], cv2.resize(predicted_img4, (400, 500)))
cv2.imshow(subjects[5], cv2.resize(predicted_img5, (400, 500)))

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()





