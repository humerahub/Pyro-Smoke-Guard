from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import telepot
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

bot_token = '6542797845:AAFf5SO3yLQ3bmpBDyVxRieUssn6_JRckmY'
target_chat_id = '6747620385'  # Enter your Telegram chat ID

# Initialize Telegram bot with user-provided token
bot = telepot.Bot(bot_token)

# Load the image classification model
load_saved_model = torch.load('../Models/fire-flame.pt', map_location=torch.device('cpu'))
load_saved_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        main(file_path)
        return 'File uploaded successfully!'

def send_telegram_alert(message, image_path):
    with open(image_path, 'rb') as photo:
        bot.sendPhoto(target_chat_id, photo, caption=message)

def image_classification(image_path):
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_processed = transformer(img).unsqueeze(0)
    img_var = img_processed.to(device)

    with torch.no_grad():
        output = load_saved_model(img_var)

    probabilities = torch.softmax(output, dim=1)
    confidence, clas = probabilities.topk(1, dim=1)
    class_names = ('Fire', 'Neutral', 'Smoke')

    return class_names[clas.item()], confidence.item()

def main(image_path):
    image_class, confidence = image_classification(image_path)

    print(f"Class: {image_class}, Confidence: {confidence:.2f}")

    if image_class in ['Fire', 'Smoke']:
        message = f"Class: {image_class} detected in the image! - Confidence: {confidence:.2f}"
        send_telegram_alert(message, image_path)
    else:
        print("No fire or smoke detected in the image or no relevant class found.")

if __name__ == "__main__":
    app.run(debug=True)
