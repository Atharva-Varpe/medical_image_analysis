# Medical Image Analysis for Disease Detection

A deep learning project using convolutional neural networks (CNNs) for automated medical image classification to detect COVID-19 and Viral Pneumonia from chest X-ray images. The project leverages transfer learning with the ResNet-50 architecture to achieve high accuracy and uses Grad-CAM for model interpretability.

## Key Features

- **CNN Architecture**: Built a robust CNN using transfer learning with a pre-trained ResNet-50 model.
- **High Accuracy**: Achieved a test accuracy of **71.78%** in classifying chest X-rays into 'COVID', 'Normal', and 'Viral Pneumonia' categories.
- **Data Augmentation**: Implemented a data augmentation pipeline with `tf.keras.layers` to handle imbalanced datasets and improve model generalization.
- **Comprehensive Preprocessing**: Processed over 15,000 medical images with a pipeline that includes resizing, normalization, and batching.
- **Model Interpretability**: Applied Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the regions of interest that the model focuses on for diagnosis.
- **Robust Evaluation**: Utilized k-fold cross-validation to ensure the model's performance is stable and reliable across different subsets of data.

## Technologies Used

- **Core Libraries**: TensorFlow/Keras, NumPy, Pandas
- **Model Architecture**: ResNet-50, Transfer Learning
- **Image Processing**: OpenCV, PIL
- **Model Evaluation**: Scikit-learn, K-Fold Cross-Validation
- **Visualization**: Matplotlib, Seaborn, Grad-CAM

## Dataset

The model was trained on the **COVID-19 Radiography Database**, which contains chest X-ray images for three classes: COVID-19, Viral Pneumonia, and Normal.

- **Image Size**: `(224, 224)`
- **Batch Size**: `32`
- **Classes**: `['COVID', 'Normal', 'Viral Pneumonia']`

## Model Performance

The ResNet-50 based model achieved the following performance on the test set:
- **Test Accuracy**: 71.78%
- **Test Precision**: 72.25%
- **Test Recall**: 70.76%

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/medical-image-analysis.git](https://github.com/your-username/medical-image-analysis.git)
    cd medical-image-analysis
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset**:
    Download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place it in the project directory. Update the `DATASET_PATH` variable in the notebook to point to the correct location.

4.  **Run the Jupyter Notebook**:
    Launch Jupyter Notebook and open `medical_image_analysis.ipynb` to train the model and evaluate its performance.
    ```bash
    jupyter notebook medical_image_analysis.ipynb
    ```
