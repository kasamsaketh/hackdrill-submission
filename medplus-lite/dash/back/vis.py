from flask import Flask, request, jsonify
from model_loader import get_medical_response
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
        logger.info(f"Received question: {user_input}")
        
        if not user_input:
            return jsonify({"response": "Please send a message."}), 400

        # Get AI response
        response = get_medical_response(user_input)
        logger.info(f"Generated response length: {len(response)} characters")
        
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "response": f"Sorry, I encountered an error while processing your question. Please try again or rephrase your question."
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "AI Medical ChatBot is running"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Medical ChatBot API is running",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health"
        }
    })

if __name__ == "__main__":
    print("Starting AI Medical ChatBot...")
    print("Make sure to install dependencies: pip install torch transformers flask flask-cors")
    print("API will be available at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)