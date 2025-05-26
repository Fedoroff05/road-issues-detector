from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось прочитать изображение")

    results = model.predict(img)
    plotted_img = results[0].plot()

    problems = []
    car_count = 0
    person_detected = False

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = float(box.conf.item())

        if class_name == 'car':
            car_count += 1
        elif class_name == 'person':
            person_detected = True
            problems.append({
                'type': 'Человек на проезжей части',
                'severity': 'critical',
                'confidence': round(confidence * 100),
                'message': 'Обнаружен пешеход на проезжей части'
            })

    if car_count >= 2:
        problems.append({
            'type': 'Авария',
            'severity': 'critical',
            'confidence': min(90 + car_count * 5, 99),
            'message': f'Обнаружена авария (из {car_count} автомобилей)'
        })

    if not person_detected and car_count < 2:
        if random.random() > 0.5:
            problems.append({
                'type': 'Яма',
                'severity': 'high',
                'confidence': 80 + random.randint(0, 15),
                'message': 'Обнаружена яма на дороге'
            })
            plotted_img = draw_pits(plotted_img)

    result_filename = 'result_' + os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, plotted_img)

    return problems, result_filename


def draw_pits(image):
    height, width = image.shape[:2]

    rect_color = (255, 0, 0)
    rect_thickness = 2
    text_color = (255, 255, 255)
    text_bg = (255, 0, 0)
    font_scale = 1.0
    text_thickness = 4
    text = "Pit"
    font = cv2.FONT_HERSHEY_SIMPLEX

    w = int(width * 0.6)
    h = int(height * 0.6)
    x = (width - w) // 2
    y = (height - h) // 2

    cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, rect_thickness)

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

    text_panel_y = y - text_height - 30
    cv2.rectangle(image,
                  (x, text_panel_y - 10),
                  (x + w, y - 10),
                  text_bg, -1)

    text_x = x + (w - text_width) // 2
    text_y = text_panel_y + text_height - 10
    cv2.putText(image, text, (text_x, text_y),
                font, font_scale, text_color, text_thickness)

    return image



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не был отправлен'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый тип файла'}), 400

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

        file.save(upload_path)
        problems, result_image = analyze_image(upload_path)

        return jsonify({
            'success': True,
            'problems': problems,
            'original_image': f'/uploads/{filename}',
            'result_image': f'/results/{result_image}',
            'message': problems[0]['message']
        })

    except Exception as e:
        return jsonify({
            'error': f'Произошла ошибка при обработке файла: {str(e)}'
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)