🧠 Brain Tumor Detection & Classification Using Deep Learning
📌 Project Overview

This project presents a Deep Learning–based Brain Tumor Detection and Classification System using MRI images.
The system can:

Detect whether a brain tumor is present

Classify the tumor into Glioma, Meningioma, Pituitary Tumor, or No Tumor

Provide confidence score

Display medical insights such as affected brain region, possible symptoms, and treatment options (informational)

The model is deployed as a user-friendly web application using Streamlit.

🎯 Objectives

To automate brain tumor detection using MRI images

To classify brain tumors into different types

To assist understanding through visual and medical insights

To demonstrate real-world application of CNNs in healthcare

🧠 Tumor Classes Supported

Glioma

Meningioma

Pituitary Tumor

No Tumor

🏗️ System Architecture

MRI Image Upload

Image Preprocessing (Resize, Normalize)

CNN-based Feature Extraction

Multi-Class Classification (Softmax)

Result Display with Confidence & Medical Insights

🧪 Dataset Used

Brain Tumor MRI Dataset (Kaggle – masoudnickparvar)

MRI images categorized into:

Glioma

Meningioma

Pituitary

No Tumor

The dataset was reorganized into class-wise folders and split internally for training and testing.

⚙️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Streamlit

GitHub

Streamlit Community Cloud

🤖 Model Details

Model Type: Convolutional Neural Network (CNN)

Input Size: 128 × 128 RGB images

Output Layer: Softmax (4 classes)

Loss Function: Categorical Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

🌐 Web Application Features

Simple & clean user interface

Upload MRI image

Displays:

Tumor detection result

Tumor type

Confidence score

Affected brain region

Possible symptoms

Common treatment options

Includes medical disclaimer

⚠️ Disclaimer

This application is developed for educational purposes only.
It is not intended for medical diagnosis or clinical use.

🏆 Conclusion

This project demonstrates how deep learning and computer vision can assist in early brain tumor detection and classification, highlighting the potential of AI in healthcare applications.

👩‍💻 Author

Donthireddy Rakshitha
CSE / Vignan Institute Of Technology And Science
3rd-Year
