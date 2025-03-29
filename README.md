## Project1_A24
Skin cancer diagnosis is a critical field in medical research, aiming to enhance early detection and treatment outcomes. Leveraging advanced machine learning and deep learning techniques, this project develops an accurate and efficient skin disease classifier. The HAM10000 dataset, a benchmark dataset in dermoscopic imaging, is used to train, validate, and test the performance of the model.


## Team Members

* Justyna Dobersztajn
* Diba Dabiransari
* Aleeza Azad


## Project Management and Quality Development

We adopt a hybrid methodology combining **Agile Project Management** and **Six Sigma** principles for efficiency and quality assurance. Agile ensures iterative development, flexibility, and collaboration, while Six Sigma's DMAIC framework focuses on quality control and continuous improvement.

### Quality Development Practices
To deliver a robust and reliable solution, we utilize:
- **Test-Driven Development (TDD):** Writing tests before implementation to ensure functionality.
- **Test Automation:** Automating tests to validate code and detect regressions efficiently.
- **Continuous Integration (CI):** Ensuring code integrity through automated builds and tests.
- **Peer Review and Pair Programming:** Promoting collaboration, knowledge sharing, and adherence to coding standards.
- **Refactoring:** Simplifying and optimizing code for maintainability.

This approach ensures a flexible yet structured workflow, delivering high-quality outcomes in skin disease diagnosis and classification.


## About Our Dataset
The **HAM10000 ("Human Against Machine with 10000 Training Images")** dataset is a comprehensive collection of dermoscopic images of common pigmented skin lesions. It contains **10,015 high-quality dermoscopic images** categorized into **seven classes** representing various skin conditions:

1. **Melanocytic nevi (nv)**
2. **Melanoma (mel)**
3. **Benign keratosis-like lesions (bkl)**
4. **Basal cell carcinoma (bcc)**
5. **Actinic keratoses and intraepithelial carcinoma (akiec)**
6. **Vascular lesions (vasc)**
7. **Dermatofibroma (df)**

The dataset was acquired from multiple sources, ensuring diversity in lesion types, skin types, and imaging techniques. It is publicly available and widely used in dermatological research.

![image](https://github.com/user-attachments/assets/925b84e2-3388-4e5f-ad32-0458e4414d0f)

### Dataset Diversity

The HAM10000 dataset was acquired from multiple sources, ensuring diversity in lesion types, skin types, and imaging techniques. The images were sourced from **clinical dermatology datasets**, as well as **public image repositories**, which contributed to the variety of skin conditions and patient demographics represented. Key features of the dataset include:

- **Lesion Variety**: The dataset covers a wide range of skin conditions, including both malignant (e.g., melanoma) and benign (e.g., melanocytic nevi, dermatofibromas) lesions. This ensures that machine learning models trained on this dataset can generalize across different types of lesions.
  
- **Diverse Skin Types**: The dataset includes images representing various **skin tones** and **ethnicities**, ranging from light to dark skin, which is crucial for developing algorithms that work effectively across diverse populations.

- **Imaging Conditions**: Images in the dataset were captured using different **dermoscopic devices** and under varying **lighting conditions**. This variability helps ensure that models can recognize skin lesions accurately regardless of imaging conditions.

- **Age Range**: The dataset represents a wide range of **age groups**, from children to elderly individuals, making it a valuable resource for age-inclusive skin disease detection.

- **Image Quality**: Each image in the dataset is of high quality, with clear dermoscopic views of the lesions, allowing for detailed feature extraction and model training.


## Pre-processing Stages


## Training Models


## Evaluating Models


## Deployment


## Libraries Used in this Project

```
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

## Version History

