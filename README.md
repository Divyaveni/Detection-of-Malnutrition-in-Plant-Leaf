# Detection-of-Malnutrition-in-Plant-Leaf



# 🌾 Detection of Malnutrition in Rice Plant Leaves using CNN

This project focuses on detecting nutrient deficiencies—Nitrogen (N), Phosphorus (P), and Potassium (K)—in **rice plant leaves** using **Image Processing** and **Convolutional Neural Networks (CNNs)**. The study explores traditional CNN architectures as well as transfer learning and ensemble methods for improved classification performance.

---

## 📌 Problem Statement

Nutrient deficiencies in crops can lead to severe losses in yield and quality. Early and accurate detection helps mitigate damage and optimize resource use. This project aims to classify rice leaf images into three categories based on malnutrition type using deep learning models.

---

## 🧠 Proposed Model

A **custom CNN architecture** was designed after experimentation with transfer learning models (VGG-16, ResNet-50, Inception). The CNN comprises:

* 3 Convolutional + MaxPooling blocks
* Flatten and Dropout layers (dropout rate = 0.5)
* Dense layers with 512 and 128 units
* Final Dense layer for classification
* **Attention mechanism** for improved focus on key regions

An **ensemble model** of three CNNs was also evaluated for performance boost.

---

## 🖼️ Dataset

* **Source:** [Kaggle - Nutrient Deficiency in Rice](https://www.kaggle.com/datasets/guy007/nutrientdeficiencysymptomsinrice)
* **Classes:**

  * Nitrogen Deficiency (N)
  * Phosphorus Deficiency (P)
  * Potassium Deficiency (K)
* **Preprocessing:**

  * Resized to 256x256 pixels
  * Converted from RGB to HSV
  * Applied `MedianBlur()` for noise reduction
  * Equal number of images per class (330)
  * Augmentation: rotation, zoom, flip, etc.

---

## 🛠️ Methodology

### 1. Data Augmentation

Applied transformations to expand dataset and reduce overfitting:

* Horizontal/vertical flips
* Rotation and zoom
* Brightness/contrast changes

### 2. Model Architectures Tested

#### 🔁 Transfer Learning Models

| Model     | Test Accuracy | Loss   |
| --------- | ------------- | ------ |
| VGG-16    | 63.0%         | 0.8699 |
| Inception | 74.7%         | 1.2955 |
| ResNet-50 | 48.2%         | 1.3374 |

#### 🧱 Custom CNN Architecture

| Model           | Train Accuracy | Test Accuracy | Loss   |
| --------------- | -------------- | ------------- | ------ |
| CNN (50 Epochs) | 82.9%          | 76.2%         | 0.5638 |
| CNN (80 Epochs) | 89.3%          | 81.3%         | 0.5058 |

#### 🧩 Ensemble CNN Model

| Model                | Train Accuracy | Test Accuracy | Loss   |
| -------------------- | -------------- | ------------- | ------ |
| Ensemble (50 Epochs) | 93.1%          | 79.6%         | 0.5106 |
| Ensemble (80 Epochs) | 94.5%          | 82.5%         | 0.4865 |

---

## 🧪 Experimental Results

* **Best performance** achieved with the **ensemble CNN model** (80 epochs).
* **Custom CNN** outperformed all transfer learning models.
* Confusion matrix and training graphs highlight improved class separation and accuracy with ensemble learning.

---

## 🧠 Key Features

* Image noise reduction using `MedianBlur`
* HSV color space conversion
* Balanced dataset with class-wise augmentation
* Ensemble learning for robust predictions
* Attention layers to highlight critical leaf regions

---

## 🧾 Conclusion

This project demonstrates the successful use of **deep learning**, especially **CNNs and ensemble models**, for detecting malnutrition types in rice leaves. Transfer learning models underperformed due to domain mismatch, while the custom CNN architecture and ensemble approach yielded promising results.

---

## 🔭 Future Scope

* Extend classification to micronutrient deficiencies and water content analysis.
* Improve model architecture for real-time inference.
* Build and deploy a **web application** for farmers/agronomists to upload leaf images and get instant analysis.

---

## 📚 References

1. Talukder & Sarkar, SSRN: [Link](https://ssrn.com/abstract=4217286)
2. Detection of Plant Leaf Nutrients using CNN: [IJNAA Journal](http://dx.doi.org/10.22075/IJNAA.2021.5194)
3. Kaggle Dataset: [Nutrient Deficiency in Rice](https://www.kaggle.com/datasets/guy007/nutrientdeficiencysymptomsinrice)
4. CNN Overview: Yamashita et al., *Insights into Imaging*, 2018.
5. CNN Model Architecture Reference: [ResearchGate](https://www.researchgate.net/figure/Proposed-adopted-Convolutional-Neural-Network-CNN-model_fig2_332407214)
6. Ensemble Learning Guide: [Neptune.ai Blog](https://neptune.ai/blog/ensemble-learning-guide)

