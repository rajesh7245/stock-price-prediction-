# scripts/api/index.py
from flask import Flask, jsonify
import sys
import os

# Ensure s.py is imported from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app import app  # import the Flask app from s.py

@app.route("/api")
def api_home():
    return jsonify({
        "message": "API is live!",
        "endpoints": ["/predict?ticker=AAPL&period=6mo", "/"]
    })

# Vercel uses 'app' as the entry point
handler = app
