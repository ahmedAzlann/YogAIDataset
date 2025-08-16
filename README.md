YogAIDataset
YogAIDataset is a dataset and model training project for Yoga Pose Recognition using Python, OpenCV, Mediapipe, and Machine Learning/Deep Learning techniques.

🚀 Features -
Collect pose data using a webcam or external device (e.g., DroidCam, Iriun).
Store key landmarks (joints & body points) for different yoga poses.
Train classification models (Random Forest, SVM, Neural Networks, etc.) to recognize poses.
Includes scripts for data capture, preprocessing, and training.

folder structure - 

YogAIDataset/
│
├── training_scripts/        # Scripts for training/testing models
│   ├── capture_pose_data.py
│   ├── train_pose_model.py
│   └── utils.py
│
├── data/                    # Captured CSV datasets
│   ├── warrior_pose.csv
│   ├── tree_pose.csv
│   └── ...
│
├── yoga_env/                # (Virtual Environment - ignored in git)
│
└── README.md                # Project documentation

Installation - 
Clone the repository:
git clone https://github.com/ahmedAzlann/YogAIDataset.git
cd YogAIDataset


Create & activate a virtual environment (recommended) - 

python -m venv yoga_env
yoga_env\Scripts\activate   # On Windows
source yoga_env/bin/activate # On Linux/Mac

Install dependencies:
pip install -r requirements.txt

Usage - 

Capture Data
python training_scripts/capture_pose_data.py
Perform yoga poses in front of your camera to record landmarks.

Train Model
python training_scripts/train_pose_model.py

Test / Predict
python training_scripts/test_pose_model.py

About Dataset - 
The dataset consists of body landmarks extracted using Mediapipe.
Each CSV contains key points (x, y, z, visibility) for each detected body landmark across multiple yoga poses.

Tech Stack - 
Python 3.10+
OpenCV – video processing
Mediapipe – pose detection
Scikit-learn / TensorFlow – training models
Pandas, NumPy, Matplotlib – data handling & visualization 


Contributing
Pull requests are welcome! If you’d like to add more yoga poses or improve the training pipeline, feel free to fork and submit a PR.

📜 License
This project is licensed under the MIT License.
