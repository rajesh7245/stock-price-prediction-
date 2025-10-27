from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from your Python app on Vercel!"

if __name__ == "__main__":
    app.run()
