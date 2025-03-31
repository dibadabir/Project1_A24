# Project1_A24

Skin cancer diagnosis is a critical field in medical research, aiming to enhance early detection and treatment outcomes. Leveraging advanced machine learning and deep learning techniques, this project develops an accurate and efficient skin disease classifier. The HAM10000 dataset, a benchmark dataset in dermoscopic imaging, is used to train, validate, and test the performance of the model.

---

## 👥 Team Members

- Justyna Dobersztajn  
- Diba Dabiransari  
- Aleeza Azad  

---

## 📋 Project Management and Quality Development

We adopt a hybrid methodology combining **Agile Project Management** and **Six Sigma** principles for efficiency and quality assurance. Agile ensures iterative development, flexibility, and collaboration, while Six Sigma's DMAIC framework focuses on quality control and continuous improvement.

<p align="center">
  <img src="https://media.licdn.com/dms/image/v2/D4D12AQHmPnR839fivw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1683634174404?e=2147483647&v=beta&t=9Ds4fAfQf6Uu3EBgk3aYKPiPLX1ALEwyrbeqJW0uYhY" alt="Image 1" width="45%" style="margin-right: 10px;"/>
  
  <img src="https://media.licdn.com/dms/image/D5612AQFoCKcAHtMjgg/article-cover_image-shrink_600_2000/0/1696230326775?e=2147483647&v=beta&t=uGKmCkizvd_C9Rl5d9Jsru1b6cIIm6Sdr3uEvnA7Ni4" alt="Image 2" width="45%"/>
</p>





### 🔧 Quality Development Practices

To deliver a robust and reliable solution, we utilize:

- **Test-Driven Development (TDD)**: Writing tests before implementation to ensure functionality.  
- **Test Automation**: Automating tests to validate code and detect regressions efficiently.  
- **Continuous Integration (CI)**: Ensuring code integrity through automated builds and tests.  
- **Peer Review and Pair Programming**: Promoting collaboration, knowledge sharing, and adherence to coding standards.  
- **Refactoring**: Simplifying and optimizing code for maintainability.  

This approach ensures a flexible yet structured workflow, delivering high-quality outcomes in skin disease diagnosis and classification.

---

## 📊 About Our Dataset

### 1. Custom Dataset – Normal Skin

To create a more balanced and comprehensive training dataset, we developed our own **Normal Skin Dataset**. This dataset consists of:

- **320 original images** captured using smartphone cameras under various lighting and environmental conditions.
- **Data Augmentation Techniques** applied to enhance generalization and prevent overfitting:
  - Rotation  
  - Flipping (horizontal/vertical)  
  - Brightness and contrast adjustment  
  - Addition of Gaussian noise  

➡️ **This increased the dataset to 1,000 images of normal, lesion-free skin.**

<p align="center">
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2017.10.01.png" alt="Alt text" style="width:50%;">
</p>

### 2. HAM10000 Dataset

The **HAM10000** ("Human Against Machine with 10000 Training Images") dataset is a comprehensive collection of dermoscopic images of common pigmented skin lesions. It contains **10,015 high-quality dermoscopic images** categorized into seven classes:

- Melanocytic nevi (`nv`)  
- Melanoma (`mel`)  
- Benign keratosis-like lesions (`bkl`)  
- Basal cell carcinoma (`bcc`)  
- Actinic keratoses and intraepithelial carcinoma (`akiec`)  
- Vascular lesions (`vasc`)  
- Dermatofibroma (`df`)  

<p align="center">
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2017.16.02.png" alt="Alt text" style="width:50%;">
</p>

---

## 🧪 Pre-processing Stages

To prepare our data for effective training, we performed the following preprocessing steps:

- **Grouping and Labeling**: Categorizing images into specific lesion types and adding corresponding labels.

<p align="center">
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2017.23.06.png" alt="Image 1" width="45%" style="margin-right: 10px;"/>
  
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2017.24.01.png" alt="Image 2" width="45%"/>
</p>

- **Image Resizing**: Standardizing image dimensions to fit model input requirements (e.g., 224x224).  
- **Normalization**: Scaling pixel values between 0 and 1 to improve convergence.  
- **Augmentation**: Generating new training samples using real-time augmentation to increase dataset variability.  
- **Hair Removal (DullRazor Algorithm)**: Removing artifacts such as body hair which obscure lesion boundaries and affect model performance.  

<p align="center">
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2017.41.59.png" alt="Image 1" width="45%" style="margin-right: 10px;"/>
  
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2017.41.37.png" alt="Image 2" width="45%"/>
</p>

---

## 🧠 Training Models

We implemented three core models for skin lesion classification:

- **Custom CNN**: A lightweight, self-built deep learning architecture tailored to our dataset.  
- **VGG16**: A well-known deep CNN architecture, pretrained on ImageNet and fine-tuned for skin lesion classification.  
- **ResNet50**: A deep residual learning model capable of handling vanishing gradient problems in very deep networks.  

➡️ **Transfer learning** was leveraged to improve learning efficiency and accuracy by utilizing pretrained weights from large-scale datasets and adapting them to our medical imaging use case.

---

## 📈 Evaluating Models

To assess model performance, we used multiple evaluation metrics and visualizations:

- **Training and Validation Loss/Accuracy Curves**: To monitor overfitting and generalization ability.  
- **Confusion Matrix**: To visualize prediction distribution across actual vs. predicted labels.  
- **ROC Curve**: To measure the true positive rate against the false positive rate for each class.  
- **Classification Report**: Including Precision, Recall, F1-score, and Accuracy.

<p align="center">
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2018.00.19.png" alt="Image 1" width="50%" style="margin-right: 10px;"/>
  
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2018.00.36.png" alt="Image 2" width="25%"/>
</p>

<p align="center">
  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2018.00.51.png" alt="Image 2" width="35%" style="margin-right: 10px;"/>

  <img src="https://github.com/dibadabir/Project1_A24/blob/main/Images/Screenshot%202025-03-31%20at%2018.01.06.png" alt="Image 2" width="45%"/>
</p>

> Since the model is used for healthcare diagnostics, **minimizing false negatives** (i.e., high recall) is critical. Early and correct identification of malignant lesions can significantly affect treatment success and patient survival rates.

---

## 🚀 Deployment

**Deployment details coming soon...**

---

## 🧰 Libraries Used in this Project

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import VGG16, ResNet50
