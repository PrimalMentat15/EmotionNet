# EmotionNet 🎭  
A Facial Expression Recognition System using CNN and FER2013 Dataset  
> A Deep Learning model trained to classify human emotions from grayscale facial images.

---

## 🧠 Overview

**EmotionNet** is a facial expression recognition project that classifies facial emotions using a Convolutional Neural Network (CNN) trained on the **FER2013 dataset**. The model distinguishes between seven emotional categories: **Angry, Disgust, Fear, Happy, Sad, Surprise,** and **Neutral**.

This notebook-based project explores data preprocessing, model building using Keras and TensorFlow, training, evaluation, and result visualization. It is aimed at applications in emotion-aware AI systems and human-computer interaction.

---

## 🎯 Objective

To develop a deep learning pipeline that:
- Loads and preprocesses the FER2013 dataset.
- Trains a custom CNN to classify facial expressions.
- Evaluates the model using metrics and visualizations.
- Plots training performance (accuracy and loss).
- Demonstrates emotion prediction on test samples.

---

## 🚀 Features

- 📁 Loads FER2013 data from CSV
- 🧠 CNN model built using TensorFlow/Keras
- 📊 Training with validation split and progress monitoring
- 📈 Accuracy & loss visualization across epochs
- 🧪 Model evaluation with classification report and confusion matrix
- 🖼️ Random test image prediction with label output

---

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras** (for model building)
- **NumPy, Pandas** (for data handling)
- **Matplotlib, Seaborn** (for plotting)
- **Scikit-learn** (for evaluation metrics)

---


## 📊 Dataset – FER2013

- Source: [Kaggle FER2013 Dataset]([https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge/data))
- Format: CSV with pixel values and emotion labels
- Emotion Classes:
0 = Angry
1 = Disgust
2 = Fear
3 = Happy
4 = Sad
5 = Surprise
6 = Neutral

- Image size: 48x48 grayscale

---

## 📦 How to Run (via Google Colab)

1. Open the notebook: [EmotionNet.ipynb](https://colab.research.google.com/drive/11x7maRXKX2v2SU85y_MPlM0d8bnIN0tO?usp=sharing)
2. Upload the **FER2013 CSV file** from Kaggle to your Colab environment.
3. Run each cell sequentially:
 - Data loading and preprocessing
 - Model architecture definition
 - Model training
 - Evaluation
 - Predictions on test samples

---

## 📈 Results

- **Training Accuracy**: ~88.90%
- Confusion matrix and classification report generated in the notebook
- Training plots available for accuracy and loss over epochs

*Update these values with actual training results once finalized.*

---

## 🧠 Model Architecture

- Input: 48x48 grayscale image
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense → Dropout → Output (Softmax)
- Total Params: ~13M

---

## ✅ Future Scope

- Real-time webcam-based emotion detection using OpenCV
- Model deployment using Streamlit or Flask
- Training on enhanced datasets with more diversity
- Hyperparameter tuning and regularization
- Transfer learning with pretrained models (e.g., ResNet, MobileNet)

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙋‍♂️ Author

**Moksh Gupta**  
B.Tech CSE, Manipal University Jaipur 

---

## 🌐 Acknowledgments

- [Kaggle - FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- TensorFlow and Keras Community
- Online tutorials and CNN architecture references


