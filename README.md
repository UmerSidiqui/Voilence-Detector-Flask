Violence Detection System
This repository contains a Flask web application that processes uploaded video files to detect instances of violence in real-time. The application uses a retrained YOLOv8 model for object detection and identifies violent activities within video frames.

Features
Upload Video Files: Users can upload video files via the web interface.
Real-time Processing: The application processes the video and performs violence detection in real-time.
Annotated Video Stream: The video feed is displayed with annotations showing detected objects and violence alerts.

Installation
Clone the repository:

git clone <https://github.com/UmerSidiqui/voilencedetector)>
cd <repository-directory>
Install dependencies:
Ensure you have Python 3.8+ installed. Then, install the required packages:

pip install -r requirements.txt
Place the retrained YOLOv8 model:
Ensure that the YOLOv8 model (yolov8n_retrained_final.pt) is in the root directory of the project.

Create necessary directories:
The application stores uploaded videos in a directory named static/videos. This directory will be created automatically if it doesn’t exist.

Usage
Run the Flask application:

python app.py

The application will start on http://127.0.0.1:5000/.

Upload a video file:

Navigate to the home page.
Click on the upload section, choose a video file, and submit.
View the processed video:

After the video is uploaded, you will be redirected to a page that streams the processed video.
The video will display annotations indicating any detected violence.

Code Explanation
Key Components

Flask Routes:

/: Home route that handles file uploads and displays the upload form.
/stream_processing/<filename>: Route that handles video processing and displays the video stream.
/video_feed/<filename>: Route that streams the video with violence detection annotations.
Violence Detection:

The YOLOv8 model (yolov8n_retrained_final.pt) is used to detect objects in each video frame.
If any object detected corresponds to "violence," it’s flagged, and the frame is annotated.
Video Streaming:

The generate_frames function processes each frame of the video, runs inference with YOLOv8, and streams the annotated frames to the user.
