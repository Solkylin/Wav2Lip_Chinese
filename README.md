# LipSync System

## Project Introduction
This project is based on an improved Wav2Lip model, achieving synchronization between audio and video lip movements. The network structure and other aspects have been optimized to better fit Chinese sentences. It aims to enhance video production quality and viewing experience. 


## Requirements
- Python >= 3.8
- PyTorch >= 1.8
- CUDA (recommended for GPU acceleration)
- Flask (for web application deployment)
- OpenCV-python
- PIL
- Add any other missing dependencies as required

## Installation Guide

### Create Python Virtual Environment
```bash
conda create -n wav2lip python=3.8
```

### Activate Virtual Environment
```bash
conda activate wav2lip
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
Note: If you encounter dependency version conflicts, specify the correct versions based on the error messages.

## Running the Project

### Start Backend Service
```bash
python app.py
```
This step will start the Flask server to handle video uploads.

### Access Page
Open http://127.0.0.1:5002 in your browser, and upload a video for lip synchronization.

### Local Usage
```bash
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face <path to face image or video> --audio <path to audio file> --outfile <path and name for the output file>
```

## Model Training
### Data Preparation
Prepare matching video and audio files as training and validation datasets. Ensure that the videos clearly show the speaker's face. Add the data paths to the filelists/ directory under train.txt and val.txt in the format video file path|audio file path.

### Training Scripts
Choose a training script:
- Basic Wav2Lip model: `wav2lip_train.py`
- High-quality Wav2Lip model: `hq_wav2lip_train.py`
- Color synchronization network:`color_syncnet_train.py`

#### Basic Model Training
```bash
python wav2lip_train.py --data_root <dataset root directory> --checkpoint_dir <directory to save checkpoints>
```

#### High-Quality Model Training
```bash
python hq_wav2lip_train.py --data_root <dataset root directory> --checkpoint_dir <directory to save checkpoints>
```

#### Color Synchronization Network Training
```bash
python color_syncnet_train.py --data_root <dataset root directory> --checkpoint_dir <directory to save checkpoints>
```

### Hyperparameter Adjustment
Adjust hyperparameters such as learning rate, batch size, and number of training iterations by editing the scripts or using command-line arguments.

### Training Monitoring
Use TensorBoard to monitor training progress and performance metrics:
```bash
tensorboard --logdir=<checkpoint directory>
```

## Project Structure
```plaintext
.
├── app.py                        # Main file for Flask web application
├── audio.py                      # Audio processing module, including audio feature extraction
├── color_syncnet_train.py        # Script for training the color synchronization network model
├── hparams.py                    # Stores hyperparameters for model training and inference
├── hq_wav2lip_train.py           # Script for training the high-quality Wav2Lip model
├── inference.py                  # Main script for running lip synchronization inference
├── preprocess.py                 # Data preprocessing script for preparing training and testing data
├── wav2lip_train.py              # Script for training the basic Wav2Lip model
├── checkpoints/                  # Directory for storing pretrained model weights, such as wav2lip_gan.pth
├── evaluation/                   # Scripts and tools for evaluating model performance
├── face_detection/               # Face detection module for locating faces in videos
├── filelists/                    # File lists for training and testing data
├── input/                        # Input folder for storing videos and audio files to be processed
├── models/                       # Code defining Wav2Lip and related models
├── results/                      # Folder for storing output videos after lip synchronization
├── static/                       # Static files for the web application
├── temp/                         # Temporary folder for storing intermediate data during processing
├── templates/                    # HTML template files for the Flask web application
│   └── index.html                # Frontend page
└── uploads/                      # Directory for storing user-uploaded files in the web application
```

## Notes
- The project is configured to use GPU by default. To run on CPU, modify the device configuration in `inference.py`.
- When training models, ensure the dataset paths are correct and adjust model parameters as needed.
- For local or server use, make sure the paths to the provided audio and video files are correct and the checkpoint_path points to a valid pretrained model weight file.
- The web application allows users to upload video and audio files for lip synchronization processing. Ensure the uploads/ directory has enough space to store user-uploaded files.
- pretrained model checkpoints\wav2lip_gan.pth "https://drive.google.com/file/d/1IyDsmXJ_W8n3eAVaLjJBcuz-uvYJo9Gw/view?usp=sharing"
