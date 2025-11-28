# Drone vs Bird Detection System  
### *Real-Time Object Detection & Image Classification Using YOLO + ResNet*

---

## **Overview**

This project builds a **two-stage intelligent detection system** that can:  
1. **Detect and localize flying objects** (birds or drones) in real time using **YOLO**.  
2. **Classify the detected object** using a **ResNet-based image classifier** to ensure high accuracy.

This hybrid system solves a real-world security problem:  
> **Airports, defense zones, and critical infrastructure often face drone intrusions that look visually similar to birds. Misidentification can cause false alarms, unsafe responses, and operational disruption.**

This model helps automate this detection reliably.

---

## **Problem Statement**

Modern airspace security systems must:  
- **Differentiate drones from birds accurately**,  
- **In real time**,  
- **With limited labeled data**,  
- **Under varying lighting and weather conditions**.

False positives → unnecessary alerts, wasted resources  
False negatives → massive safety & security risks  

This project tackles that challenge using a robust multi-model AI pipeline.

---

## **Solution Architecture**

### **1. YOLO Object Detector**
- Detects objects in images and localizes them with bounding boxes.
- Outputs bounding boxes + class probabilities (bird/drone).

### **2. ResNet Classification Model**
- Takes cropped YOLO detections as input.
- Performs fine-grained classification.
- Solves the “bird or drone?” ambiguity with high accuracy.

Together, they form a **precision-optimized, real-time surveillance model**.

---

## **ResNet Classification Model (Final)**

### **Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| Bird | 0.96 | 0.96 | 0.96 | 121 |
| Drone | 0.95 | 0.95 | 0.95 | 94 |
| **Overall Accuracy** | **0.95** | | | **215 samples** |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | |

**Takeaway:**  
The classifier achieves **95% accuracy**, making it extremely reliable for distinguishing drones from visually similar birds. This significantly reduces false alarms in critical environments.

---

## **YOLO Detection Model — Key Training Metrics**

### **Final Epoch Metrics (Epochs 6–10)**

| Epoch | Precision | Recall | mAP50 | mAP50–95 | Train Box Loss | Val Box Loss |
|-------|-----------|--------|--------|-----------|------------------|----------------|
| 8 | 0.802 | 0.651 | 0.723 | 0.390 | 1.38 | 1.59 |
| 9 | 0.815 | 0.726 | 0.794 | 0.466 | 1.33 | 1.48 |
| **10** | **0.845** | **0.719** | **0.805** | **0.466** | **1.26** | **1.47** |

**Takeaway:**  
- Precision improves steadily → fewer false detections  
- mAP50 reaches **0.80+**, meaning YOLO localizes objects quite accurately  
- Loss consistently decreases → model is learning effectively  

YOLO is **well-trained for real-time drone/bird detection**, even with dataset constraints.

---

## **Why This System Works**

### **1. YOLO handles detection**
- Finds objects fast  
- Works well with small flying objects  
- Operates in real time

### **2. ResNet handles classification**
- Distinguishes drones vs birds with **95% accuracy**  
- Reduces YOLO misclassification  
- Handles subtle shape differences

### **3. Combined System = High Precision + High Certainty**
Perfect for environments where **every detection matters**.

---

## **How to Run the Streamlit App**

```bash
streamlit run app.py
```

Upload an image -> Choose task -> YOLO detects -> ResNet classifies -> Output displayed instantly

> Note: Run the classification_model.ipynb and object_detection_model.ipynb to train the models

---

## **Real-World Value**

This project provides a deployable vision system that can be integrated into:

### **Airport Surveillance Systems**
Detect drones early & prevent runway shutdowns.

### **Military / Border Perimeter Monitoring**
Differentiate harmless birds from potentially hostile UAVs.

### **Wildlife Protection**
Track bird activity while filtering drone noise.

### **Stadium / Event Security**
Prevent unauthorized drone recording or attacks.

This project successfully delivers a practical, high-accuracy, real-time AI solution for drone vs bird detection. By combining the speed of YOLO with the precision of ResNet, the system significantly reduces false alarms and enhances safety in security-critical applications such as airports, borders and event surveillance.

It demonstrates a fully deployable end-to-end pipeline—from dataset creation to model training to application deployment—solving an important real-world problem.