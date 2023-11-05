from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, MarianMTModel, MarianTokenizer, \
    GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer
import re
import logging


# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # Load the GPT-3 model and tokenizer (More powerful than GPT2)
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Initialize the sentiment analysis pipeline with the Twitter-roberta-base-sentiment model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Dynamic blacklisting of harmful prompts
blacklisted_prompts = ["<script>", "<html>", "javascript:", "onload", "onerror"]

# Regular Expression for dynamic blacklisting
dynamic_blacklist_regex = re.compile(r'|'.join(map(re.escape, blacklisted_prompts)), re.IGNORECASE)


# Function to combat prompt injection
def clean_text(text):
    cleaned_text = re.sub(r"\b([A-Z']{2,})\b", r" \1 ", text)
    return cleaned_text


# Function for Named Entity Recognition (NER) and data validation
def extract_entities(text):
    # For simplicity, let's assume we are extracting names, emails, and phone numbers
    names = re.findall(r"\b[A-Z][a-z]+\b", text)
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    phone_numbers = re.findall(r"\b\d{10}\b", text)

    # Bsic data validity checks for email and phone number
    valid_emails = [email for email in emails if re.match(r"[^@]+@[^@]+\.[^@]+", email)]
    valid_phone_numbers = [number for number in phone_numbers if len(number) == 10]

    return {"names": names, "emails": valid_emails, "phone_numbers": valid_phone_numbers}


def validate(data):
    errors = []
    if not data.get('name') or not re.match(r'^[a-zA-Z ]+$', data['name']):
        errors.append("Invalid name. Name should only contain alphabets and spaces.")
    if not data.get('email') or not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
        errors.append("Invalid email address.")
    if not data.get('phone') or not re.match(r'^\d{10}$', data['phone']):
        errors.append("Invalid phone number. Phone number should be 10 digits.")

    if not errors:
        return {"status": "success"}
    else:
        return {"status": "error", "errors": errors}


# Function for prompt injection check
def check_prompt_injection(user_input):
    if dynamic_blacklist_regex.search(user_input):
        # Log the detected malicious input
        print(f"Detected potential prompt injection in user input: {user_input}")
        logging.warning(f"Detected potential prompt injection in user input: {user_input}")
        return True
    return False


# Function for language translation
def translate_text(text, src_lang, tgt_lang):
    translator = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = translator.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text


def chatbot():
    shared_info = {"name": False, "email": False, "phone": False}

    greetings_regex = re.compile(r"\b(?:hello|hi|hey|greetings|good morning)\b", re.IGNORECASE)

    while True:
        user_input = input("User: ")
        cleaned_input = clean_text(user_input)

        # Check for prompt injection
        if check_prompt_injection(cleaned_input):
            print("Chatbot: I'm sorry, I can't process your request.")
            continue

        # Check for any request to update or correct data
        if "update" in cleaned_input.lower() or "correct" in cleaned_input.lower() or "change" in cleaned_input.lower():
            for key in shared_info.keys():
                if key in cleaned_input.lower():
                    entity = key
                    print(f"Chatbot: Sure, please provide your new {entity}:")
                    new_value = input(f"User: New {entity}: ")
                    shared_info[key] = new_value
                    print(f"Chatbot: Your {entity} has been updated to {new_value}. Is there anything else you'd like to discuss?")
                    break
            continue

        # Perform sentiment analysis
        sentiment = sentiment_analyzer(cleaned_input)[0]
        entities = extract_entities(cleaned_input)

        positive_responses = ["great", "fantastic", "wonderful", "amazing", "awesome", "excellent", "great day", "terrific", "superb", "good", "happy", "delighted", "pleased", "joyful"]

        if "email" in cleaned_input.lower() and not shared_info["email"]:
            print("Chatbot: I completely understand your concern. Your email is safe with us and will only be used for sending updates on our services and exclusive offers. Is there anything specific you're concerned about?")
            shared_info["email"] = True
        elif "phone" in cleaned_input.lower() and not shared_info["phone"]:
            print("Chatbot: Your privacy is important to us. Rest assured, we take all necessary precautions to protect your data. If you're not comfortable sharing your phone, is there another way we can stay in touch with you?")
            shared_info["phone"] = True
        elif "name" in cleaned_input.lower() and not shared_info["name"]:
            print("Chatbot: No worries, you don't have to share your name if you're not comfortable with it. So, what brings you here today? Is there anything in particular you'd like to know or discuss?")
            shared_info["name"] = True
        elif any(word in cleaned_input.lower() for word in positive_responses):
            if not shared_info["email"] and not shared_info["phone"]:
                print("Chatbot: That's fantastic to hear! By the way, I have my birthday party coming up in a week, and I'd like to invite you. Could you please share your email or phone number so I can send you the invite?")
            elif not shared_info["email"]:
                print("Chatbot: I'm glad you're feeling positive! If you don't mind, could you share your email with me so I can keep you updated?")
            elif not shared_info["phone"]:
                print("Chatbot: That's great! Could you please share your phone number so I can reach out to you in the future?")
            else:
                print("Chatbot: How can I assist you further?")
        elif re.search(greetings_regex, cleaned_input):    # Handle user greetings
            print("Chatbot: Hello! It's great to see you. How can I assist you today?")
        else:
            response = generate_response(cleaned_input, shared_info)
            # Translate the response to the original language if the user input was not in English
            if not user_input.isascii():
                response = translate_text(response, src_lang="en", tgt_lang="auto")
            print(response)


# Function to generate a response based on user input
def generate_response(user_input, shared_info):
    input_ids = tokenizer.encode(user_input, return_tensors="pt", max_length=20, truncation=True)
    output = model.generate(input_ids, max_length=20, num_return_sequences=1, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if shared_info["email"]:
        if "email" in user_input.lower() and "@" in user_input:
            response = f"Chatbot: Thank you for sharing your email! We'll keep you updated on all our latest offers and news. Is there anything else you'd like to share or discuss?"
        elif shared_info["name"]:
            if any(word in user_input.lower() for word in ["name", "call me"]):
                response = f"Chatbot: That's a lovely name! It's great to have you here, {shared_info['name']}! How can I assist you further?"
                match = re.search(r'call me (\w+)', user_input.lower())
                if match:
                    new_name = match.group(1)
                    shared_info["name"] = new_name
                    response = f"Chatbot: Ok {shared_info['name']}, sure thanks for being so transparent with me, well {shared_info['name']} would you like to know more about the query you had ?"
            else:
                response = f"Chatbot: {generated_text}"
        elif shared_info["phone"]:
            if any(word in user_input.lower() for word in ["phone", "number"]):
                response = f"Chatbot: Thank you for sharing your phone number! We'll ensure to keep you posted via text messages. Is there anything else you'd like to discuss?"
            else:
                response = f"Chatbot: {generated_text}"
        else:
            response = f"Chatbot: {generated_text}"
    else:
        response = f"Chatbot: {generated_text}"

    return response


# Run the chatbot
chatbot()
