# Brain Tumor Image Classification using CNN

![Python](https://skillicons.dev/icons?i=python) ![TensorFlow](https://skillicons.dev/icons?i=tensorflow) ![Keras](https://skillicons.dev/icons?i=keras) ![NumPy](https://skillicons.dev/icons?i=numpy) ![Matplotlib](https://skillicons.dev/icons?i=matplotlib)

**Course:** Deep Learning 
**Dataset:** Brain Tumor Images  
**Year:** 2025  



## About This Project
This project focuses on classifying brain tumor images using Convolutional Neural Networks (CNN). The workflow includes Exploratory Data Analysis (EDA), preprocessing, building a baseline AlexNet CNN, modifying the architecture for performance improvement, and evaluating the model using multiple metrics.


## Dataset Features
The dataset contains medical brain MRI images, categorized by tumor type. Important characteristics include:
- Color histograms per category  
- Aspect ratio and resolution  
- Variability in lighting, angles, occlusion, etc.



## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Examined dataset distribution and visual characteristics  
- Computed histograms, aspect ratios, and resolution statistics  
- Handled anomalies and preprocessed images (resizing, normalization, augmentation)  
- Split dataset into Train, Validation (15% of Train), and Test sets  

### 2. Baseline Model
- Implemented **AlexNet CNN architecture manually** (no pre-trained models)  
- Adapted output layer to match dataset categories  
- Trained baseline model for at least 10 epochs

### 3. Model Modifications
- Applied modifications such as **Dropout**, **Batch Normalization**, or alternative architectures (DenseNet, EfficientNet, etc.)  
- Performed hyperparameter tuning and trained modified models for at least 10 epochs  
- Justified modifications to improve generalization, convergence, and accuracy

### 4. Evaluation
- Evaluated models on test data using **at least 3 metrics** (e.g., Accuracy, F1-score, ROC-AUC)  
- Compared baseline vs. modified models, analyzed performance, and concluded the best model



## Technologies Used
- Python  
- TensorFlow & Keras  
- NumPy & pandas  
- Matplotlib & Seaborn  
- scikit-learn (for metrics & evaluation)  
- OpenCV / PIL (for image preprocessing)

