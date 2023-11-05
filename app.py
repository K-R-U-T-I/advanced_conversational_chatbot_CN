import os
import re
import mysql.connector
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, MarianMTModel, MarianTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)


# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize the sentiment analysis pipeline with the Twitter-roberta-base-sentiment model and tokenizer
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


# Function to combat prompt injection
def clean_text(text):
    cleaned_text = re.sub(r"\b([A-Z']{2,})\b", r" \1 ", text)
    return cleaned_text


# Function for language translation
def translate_text(text, src_lang, tgt_lang):
    translator = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translator.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text


def chatbot(user_input, shared_info=None):
    if shared_info is None:
        shared_info = {"name": False, "email": False, "phone": False}

    greetings_regex = re.compile(r"\b(?:hello|hi|hey|greetings|good morning)\b", re.IGNORECASE)

    cleaned_input = clean_text(user_input)

    # Perform sentiment analysis
    sentiment = sentiment_analyzer(cleaned_input)[0]

    positive_responses = ["great", "fantastic", "wonderful", "amazing", "awesome", "excellent", "great day", "terrific", "superb", "good", "happy", "delighted", "pleased", "joyful"]

    if "email" in cleaned_input.lower() and not shared_info["email"]:
        response = "I completely understand your concern. Your email is safe with us and will only be used for sending updates on our services and exclusive offers. Is there anything specific you're concerned about?"
        shared_info["email"] = True
    elif "phone" in cleaned_input.lower() and not shared_info["phone"]:
        response = "Your privacy is important to us. Rest assured, we take all necessary precautions to protect your data. If you're not comfortable sharing your phone, is there another way we can stay in touch with you?"
        shared_info["phone"] = True
    elif "name" in cleaned_input.lower() and not shared_info["name"]:
        response = "No worries, you don't have to share your name if you're not comfortable with it. So, what brings you here today? Is there anything in particular you'd like to know or discuss?"
        shared_info["name"] = True
    elif any(word in cleaned_input.lower() for word in positive_responses):
        if not shared_info["email"] and not shared_info["phone"]:
            response = "That's fantastic to hear! By the way, I have my birthday party coming up in a week, and I'd like to invite you. Could you please share your email or phone number so I can send you the invite?"
        elif not shared_info["email"]:
            response = "I'm glad you're feeling positive! If you don't mind, could you share your email with me so I can keep you updated?"
        elif not shared_info["phone"]:
            response = "That's great! Could you please share your phone number so I can reach out to you in the future?"
        else:
            response = "How can I assist you further?"
    elif re.search(greetings_regex, cleaned_input):    # Handle user greetings
        response = "Hello! It's great to see you. How can I assist you today?"
    else:
        response = generate_response(cleaned_input, shared_info, user_input)

    return response


# Function to generate a response based on user input
def generate_response(cleaned_input, shared_info, user_input):
    input_ids = tokenizer.encode(cleaned_input, return_tensors="pt", max_length=20, truncation=True)
    output = model.generate(input_ids, max_length=20, num_return_sequences=1, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if shared_info["email"]:
        if "email" in cleaned_input.lower() and "@" in cleaned_input:
            response = "Thank you for sharing your email! We'll keep you updated on all our latest offers and news. Is there anything else you'd like to share or discuss?"
        elif shared_info["name"]:
            match = re.search(r'call me (\w+)', user_input.lower())
            if match:
                new_name = match.group(1)
                shared_info["name"] = new_name
                response = f"Ok {shared_info['name']}, sure thanks for being so transparent with me, well {shared_info['name']} would you like to know more about the query you had ?"
            else:
                response = f"That's a lovely name! It's great to have you here, {shared_info['name']}! How can I assist you further?"
        elif shared_info["phone"]:
            if any(word in cleaned_input.lower() for word in ["phone", "number"]):
                response = "Thank you for sharing your phone number! We'll ensure to keep you posted via text messages. Is there anything else you'd like to discuss?"
            else:
                response = f"{generated_text}"
        else:
            response = f"{generated_text}"
    else:
        response = f"{generated_text}"

    return response


@app.route('/add_user_info', methods=['POST'])
def add_user_info():
    # Establish a connection with the MySQL database
    mydb = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

    data = request.get_json()
    cursor = mydb.cursor()
    sql = "INSERT INTO user_info (name, email, phone) VALUES (%s, %s, %s)"
    val = (data['name'], data['email'], data['phone'])
    cursor.execute(sql, val)
    mydb.commit()
    return jsonify({"status": "success", "message": "User information added successfully"})


@app.route('/chatbot', methods=['POST'])
def handle_chatbot_request():
    if not request.is_json:
        return jsonify({"error": "Request body must be in JSON format"})

    user_query = request.get_json()

    if 'user_input' not in user_query:
        return jsonify({"error": "Please provide a user input in the request body"})

    user_input = user_query['user_input']
    response = chatbot(user_input)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
