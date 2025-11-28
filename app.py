import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import cv2

# YOLO model
yolo_model = YOLO("saved_models/yolo_model/weights/best.pt")

# classification model (resnet)
state = torch.load("saved_models/resnet_classification_model.pt", map_location="cpu")

num_classes = state["fc.weight"].shape[0]

resnet_model = torch.hub.load("pytorch/vision", "resnet50", pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(state)
resnet_model.eval()

# classification preprocess
img_size = 224
classify_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class_names = ["Bird", "Drone"]

# bounding boxes for YOLO
def draw_boxes(image, results):
    img = np.array(image).copy()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]

            color = (0, 255, 0)
            thickness = 2

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(img,
                        f"{yolo_model.names[cls]} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)
    return img


# ui
st.title("Drone/Bird Detection & Classification")
st.write("Upload an image and run either Object Detection (YOLO) or Classification (ResNet).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # model selection
    task = st.selectbox(
        "Select Task:",
        ["Object Detection (YOLO)", "Classification (ResNet)"]
    )

    if st.button("Run Prediction"):

        if task == "Object Detection (YOLO)":
            st.subheader("YOLO Detections")

            # inference
            results = yolo_model(image)
            drawn_img = draw_boxes(image, results)

            st.image(drawn_img, caption="YOLO Output", use_container_width=True)

            # show json output
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": yolo_model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox_xyxy": [float(x) for x in box.xyxy[0]]
                    })

            st.json(detections)

        else:
            st.subheader("ResNet Classification")

            tensor = classify_transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = resnet_model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()

            st.write(f"### Prediction: **{class_names[pred_idx]}**")
            st.write(f"Confidence: **{confidence:.4f}**")


st.markdown("---")
st.write("Made using YOLOv8 + ResNet + Streamlit")
