from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'True') == 'True'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')


@app.route('/')
def index():
    return jsonify({
        'message': 'Bienvenue sur l\'API NLP',
        'status': 'ok'
    })


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5001)),
        debug=app.config['DEBUG']
    )
