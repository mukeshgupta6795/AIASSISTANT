from flask import Flask , render_template, request , jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("API_Key_GROQ")

llm = ChatGroq(
    groq_api_key=api_key,
    model="qwen/qwen3.8-27b",
    temperature=0.7,
    max_tokens=512
)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask" , methods=["POST"])
def ask():
    question = request.form.get("question")
    response = llm.invoke([
             (
                "system",
                "Act like a helpful personal assistant"
            ),
            (
                "human",
                question
            )
        ])
    answer = response.content.strip()
    #  return response.content.strip()
    return jsonify({"response" :answer}) , 200

@app.route("/summarize" , methods = ["POST"])
def summarize():
    email_text = request.form.get("email")
    prompt = f"summarize the following email in 2-3 sentence : {email_text}"
    response = llm.invoke([
             (
                "system",
                "Act like a helpful personal assistant"
            ),
            (
                "human",
                prompt
            )
        ])

    # answer = response.content.strip()
    summary = response.content.strip()
    return jsonify({"response" :summary}) , 200


if __name__ == "__main__":
    app.run(debug=True)