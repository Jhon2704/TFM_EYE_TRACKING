# Eye-Tracking Based Exam Monitoring System

## 📌 Project Description

This project is part of my **Master's Thesis (TFM)**, focused on
**biometric security through gaze detection**.\
It integrates **deep learning** techniques, data augmentation, and
real-time image processing to monitor a student's gaze during online
exams.

The system classifies eye gaze into five categories: - ⬆️ Up - ⬇️ Down -
⬅️ Left - ➡️ Right - ⭕ Center

If suspicious gaze deviations or anomalies (such as prolonged eye
closure) are detected, the system generates alerts in real time.

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   **Python 3.9**
-   **TensorFlow / Keras** → CNN custom model
-   **PyTorch** → ETH-XGaze model integration
-   **OpenCV** → Real-time video processing
-   **Tkinter** → GUI (Graphical User Interface)
-   **Scikit-learn** → Evaluation metrics (Confusion Matrix, Accuracy,
    Precision, Recall, F1-score)
-   **Matplotlib / Seaborn** → Graphs and visualizations
-   **NumPy / Pandas** → Data manipulation

------------------------------------------------------------------------

## 📂 Data Sources

-   **ETH-XGaze Dataset** → Used as baseline for comparison.
-   **Real Dataset** → Collected images from volunteers (gaze
    directions).
-   **Synthetic Dataset** → Generated with open-source tools to increase
    variability and reduce bias.

All datasets were used under appropriate **licenses**.

------------------------------------------------------------------------

## 🧪 Experiments Conducted

1.  **Real dataset with Data Augmentation** (CNN, default
    hyperparameters).
2.  **Synthetic dataset without augmentation** (CNN, default
    hyperparameters).
3.  **Grid Search on real dataset** (hyperparameter optimization).
4.  **Synthetic dataset with tuned hyperparameters**.

------------------------------------------------------------------------

## 📊 Results

-   Custom CNN model achieved **higher accuracy and inference speed**
    compared to ETH-XGaze.
-   Demonstrated that **simpler models** can outperform complex
    pre-trained ones in specific tasks.
-   System successfully detects gaze anomalies in real time.

------------------------------------------------------------------------

## 📺 Demo

The interface shows: - **Camera capture** with face and eye tracking. -
**Gaze direction arrow** (⬆️⬇️⬅️➡️). - **Alerts** when suspicious
behavior is detected. - **Log of warnings with timestamps**.

------------------------------------------------------------------------

## 📌 Contribution

This project can be expanded for: - Online exam monitoring. - Workplace
attention analysis. - Applications in driver drowsiness detection.

Pull requests and suggestions are welcome!

------------------------------------------------------------------------

## 🧑‍💻 Author

**Juan Olivan Marquina
Master's Thesis IN VIU
🔗 www.linkedin.com/in/juan-olivan-9a3210176

