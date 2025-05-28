import cv2

# Load ID card image
img_path = "test_samples/1.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load OpenCV face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Show result
if len(faces) == 0:
    print("❌ No face found. Invalid ID card.")
else:
    print(f"✅ {len(faces)} face(s) found. Photo exists in ID card.")

    # # Optional: Draw rectangle and show
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow("Detected Face", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
