#http://127.0.0.1:5002
from waitress import serve
from app import app

serve(app, host='0.0.0.0', port=5002)
