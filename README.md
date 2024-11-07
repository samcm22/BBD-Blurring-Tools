Overview
The Privacy Protection Application is a web-based tool designed to detect and redact sensitive personal information from images and videos. It leverages Optical Character Recognition (OCR) technology through Tesseract and object detection using YOLO (You Only Look Once) to identify various types of sensitive information in user-generated content.

Features
Sensitive Information Detection: Automatically detects and blurs sensitive personal information such as:
Date of Birth
ID Numbers (Alphanumeric)
Phone Numbers
Bank Names
Credit Card Numbers
Addresses
Sexual Content Keywords
Real-Time Processing: Process live camera feeds, videos, and images for privacy protection in real-time.
Customizable Detection Patterns: Regex patterns for various sensitive data types can be modified as per requirements.
Installation
Clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_directory>
Install required libraries: Ensure you have Python 3.x installed, then install the necessary libraries:

bash
Copy code
pip install opencv-python pytesseract ultralytics
Install Tesseract: Download and install Tesseract OCR from here. Update the path in the script:

python
Copy code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as needed
Download YOLO Model: Place your best.pt YOLO model file in the project directory.

Usage
Run the application by executing the following command:

bash
Copy code
python <your_script_name>.py
Select one of the following options:

Start Live Camera
Upload Video File
Upload Image File
Follow the prompts to upload files or start the camera feed for processing.

Future Development
The project is currently in its initial stage as a web-based application, with plans to develop a dedicated camera application in the future. This will enable real-time capturing and processing of images and videos for enhanced privacy protection.



Acknowledgments
OpenCV for computer vision tasks.
Tesseract OCR for optical character recognition.
YOLO for real-time object detection
