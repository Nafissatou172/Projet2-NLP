from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Import des routes
from routes import api_bp

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'True') == 'True'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Enregistrement du Blueprint principal
app.register_blueprint(api_bp)

# Route root (optionnelle si déjà dans le blueprint, mais utile pour redirection)
@app.route('/')
def root():
    return jsonify({
        'message': 'API FinChat-SN opérationnelle',
        'endpoints': ['/api', '/api/health', '/api/benchmark']
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5001)),
        debug=app.config['DEBUG']
    )
