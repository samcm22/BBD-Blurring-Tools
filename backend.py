from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import re
import pytesseract
from pytesseract import Output
from ultralytics import YOLO

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO("best.pt")

bank_name_pattern = r'\b(?:Indian Bank|SBI|RBL Bank|HDFC|Axis Bank|ICICI|PNB|Bank of Baroda)\b'
credit_card_pattern = r'\b(?:\d[ -]*?){13,16}\b'
address_pattern = r'\b\d{1,4}\s(?:[A-Za-z0-9#.-]+(?:\s(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Way|Terrace|Terr|Place|Pl|Circle|Cir|Square|Sq|Drive|Dr|Court|Ct|Close|Cl|Park|Pk|Highway|Hwy))?)\s*,?\s*(?:[A-Za-z\s]+(?:,\s*[A-Za-z\s]+)?)(?:-\d{5})?\b'
phone_number_pattern = r'\+?\d{1,4}[-.\s]?(\(?\d{1,3}?\)?[-.\s]?)?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
date_of_birth_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
id_pattern = r'\b[A-Z0-9]{5,15}\b'
name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
sexual_content_pattern = r'\b(?:sex|porn|adult|xxx|nsfw)\b'


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh


def get_bbx(img):
    return model(img)


def blur_area(img, x, y, w, h):
    roi = img[y:y + h, x:x + w]
    blurred_roi = cv2.blur(roi, (25, 25))  # Use a kernel size of 25
    img[y:y + h, x:x + w] = blurred_roi


def process_frame(img):
    processed_image = preprocess_image(img)
    data = pytesseract.image_to_data(processed_image, output_type=Output.DICT)
    n_boxes = len(data['level'])

    for i in range(n_boxes):
        text = data['text'][i].strip()

        if text:
            print(f"Detected text: '{text}' at ({data['left'][i]}, {data['top'][i]})")

        if re.search(date_of_birth_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)
        elif re.search(id_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)
        elif re.search(phone_number_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)
        elif re.search(bank_name_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)
        elif re.search(credit_card_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)
        elif re.search(sexual_content_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)
        elif re.search(address_pattern, text):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            blur_area(img, x, y, w, h)

    return img


def process_image(image_path):
    img = cv2.imread(image_path)
    processed_image = process_frame(img)
    output_path = 'output/blurred_image.jpg'
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved as: {output_path}")
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def start_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 5.0

    out = cv2.VideoWriter('output/blurred_camera_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps,
                          (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)

        cv2.imshow('Live Camera Feed', processed_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)
    process_image(image_path)
    return redirect(url_for('index'))


@app.route('/start_camera', methods=['POST'])
def run_camera():
    start_camera()
    return redirect(url_for('index'))


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
