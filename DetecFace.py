import cv2
import torch
import numpy as np
import insightface

# Load InsightFace for face detection
detector = insightface.app.FaceAnalysis(name='buffalo_l')
detector.prepare(ctx_id=0)

# Load anti-spoofing model (AtulApra's pretrained model)
class AntiSpoofModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN architecture placeholder, load actual weights below
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, pretrained=False)

    def forward(self, x):
        return self.model(x)

# For demo, let's download official weights from repo:
# (You must replace this with actual anti-spoofing model weights path)
WEIGHTS_PATH = "antispoof_model.pth"

def load_antispoof_model():
    model = torch.hub.load('biubug6/face-antispoofing', 'resnet18')  # example repo with pretrained weights
    model.eval()
    return model

model = load_antispoof_model()

def preprocess_face(face_img):
    # Resize to model input size
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = np.expand_dims(face_img, axis=0)
    face_tensor = torch.from_numpy(face_img)
    return face_tensor

def predict_antispoof(face_img):
    face_tensor = preprocess_face(face_img)
    with torch.no_grad():
        output = model(face_tensor)
        prob = torch.sigmoid(output).item()
    return prob

def main():
    image_path = "test_samples/2.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    faces = detector.get(img)
    if len(faces) == 0:
        print("No face detected!")
        return

    # Use the biggest face detected
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = img[y1:y2, x1:x2]

    score = predict_antispoof(face_crop)
    print(f"Anti-spoofing probability (live face): {score:.3f}")

    if score > 0.5:
        print("✅ Live human face detected")
    else:
        print("❌ Spoof or fake face detected")

    # Show image with box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
