import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize the classifier with model and labels
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

# Load the arrow image
imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

# Load MobileNet-SSD model for human detection
net = cv2.dnn.readNetFromCaffe('Resources/Model/MobileNetSSD_deploy.prototxt',
                               'Resources/Model/MobileNetSSD_deploy.caffemodel')

# Define the classes for the MobileNet-SSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize the classIDBin variable
classIDBin = 0

# Load all waste images
imgWasteList = []
pathFolderWaste = "Resources/Waste"
for waste_category in os.listdir(pathFolderWaste):
    category_path = os.path.join(pathFolderWaste, waste_category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                imgWasteList.append(img)

# Load all bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
for filename in os.listdir(pathFolderBins):
    img_path = os.path.join(pathFolderBins, filename)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        imgBinsList.append(img)

# Dictionary for class to bin mapping
classDic = {
    0: None,
    1: 0,
    2: 0,
    3: 3,
    4: 3,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
}

# Variable to control loop
running = True

while running:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgResize = cv2.resize(img, (454, 340))

    # Prepare image for MobileNet-SSD model
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Check for human presence
    human_present = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if CLASSES[class_id] == "person":
                human_present = True
                break

    # Load background image
    imgBackground = cv2.imread('Resources/background.png')

    # Get prediction from classifier
    prediction = classifier.getPrediction(img)
    classID = prediction[1]
    print(classID)

    if classID != 0 and not human_present:
        # Overlay waste image and arrow
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        # Get corresponding bin
        classIDBin = classDic[classID]

    # Overlay bin image if no human is detected
    if not human_present:
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    # Overlay resized webcam image
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display cool message if human is detected
    if human_present:
        overlay = imgBackground.copy()
        cv2.rectangle(overlay, (0, 0), (imgBackground.shape[1], 130), (0, 0, 255), -1)
        alpha = 0.6  # Transparency factor
        imgBackground = cv2.addWeighted(overlay, alpha, imgBackground, 1 - alpha, 0)
        cv2.putText(imgBackground, "Human Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(imgBackground, "Please show the waste.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

    # Display the final output
    cv2.imshow("Output", imgBackground)

    # Check for 'q' key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Check if stop button clicked (for example, 's' key)
    if key == ord('s'):
        running = False

# Release resources
cap.release()
cv2.destroyAllWindows()
