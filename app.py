from flask import Flask, request, send_from_directory, jsonify, url_for
import subprocess
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, Response

app = Flask(__name__, static_folder='results')

# 设置上传和结果目录的路径
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
# 确保这些目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 设置 Flask 的上传文件夹
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 定义允许的扩展名
ALLOWED_EXTENSIONS = {'mp4', 'mp3', 'wav','png','jpg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    video = request.files['video']
    audio = request.files['audio']

    # 如果用户没有选择文件，浏览器也会提交一个没有文件名的空部分
    if video.filename == '' or audio.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if video and audio and allowed_file(video.filename) and allowed_file(audio.filename):
        video_filename = secure_filename(video.filename)
        audio_filename = secure_filename(audio.filename)
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        # 检查是否是图片输入，并据此设置额外的参数
        static = False
        if video.filename.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            static = True
            result_filename = 'result_' + os.path.splitext(video_filename)[0] + '.mp4'  # 假设图片输入的结果总是视频
        else:
            result_filename = 'result_' + video_filename
            
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        
        video.save(video_path)
        audio.save(audio_path)
        
        # 构建命令行参数
        command = [
            'python', 'inference.py',
            '--checkpoint_path', 'checkpoints/wav2lip_gan.pth',
            '--face', video_path,
            '--audio', audio_path,
            '--outfile', result_path
        ]
        if static:
            command += ['--static', 'True', '--fps', '25']  # 根据需要添加 --fps 参数
        
        # 执行模型推理
                
        try:
            subprocess.run(command, check=True)
            video_url = url_for('static', filename=result_filename, _external=True)  # 修改此处以生成视频文件的URL
            return jsonify({'videoUrl': video_url})  # 返回视频URL而非直接下载
        except subprocess.CalledProcessError:
            return jsonify({'error': 'Model inference failed'}), 500
    
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
