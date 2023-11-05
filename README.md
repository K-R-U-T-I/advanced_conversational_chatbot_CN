# advanced_conversational_chatbot_CN

**Requirements**

Python 3.8
Transformers library
Additional libraries: re, logging

**Usage**

Clone the repository or download the Python script.
Ensure that the required libraries are installed.
Run the script using the following command:
`python main.py`

**Functionality**

**Prompt Injection Detection:** The chatbot checks for potential prompt injections in user input and logs any detected malicious input.
**Data Validation:** The script validates user-provided data such as names, emails, and phone numbers for their correctness and format.
**Sentiment Analysis:** It performs sentiment analysis on user inputs to determine the sentiment of the messages.
**Language Translation:** The chatbot can translate text from one language to another using the MarianMT model.
**Dynamic Blacklisting:** It prevents harmful prompts by blacklisting specific phrases.
**Named Entity Recognition:** The script extracts names, emails, and phone numbers from user inputs.

**Breakdown of the code:**

1. Importing Libraries: Importing the necessary Python libraries, including the Transformers library, regular expressions (re), and logging.
2. Model and Tokenizer Initialization: 
    It loads a pre-trained GPT-2 language model and tokenizer for generating responses.
    It also initializes a sentiment analysis pipeline using the Twitter-roberta-base-sentiment model and tokenizer.
3. Dynamic Blacklisting:
    The code defines a list of blacklisted prompts and creates a regular expression to detect and block them in user input.
4. Text Cleaning:
    A clean_text function is defined to clean the user's input text by adding spaces around words with more than two uppercase letters.
5. Named Entity Recognition and Data Validation:
    A extract_entities function extracts names, emails, and phone numbers from text.
    A validate function checks the validity of the extracted data (name, email, phone number).
6. Prompt Injection Check:
    A check_prompt_injection function checks user input for potential prompt injection by using the dynamic blacklisting regular expression.
7. Language Translation:
    A translate_text function is defined to translate text from a source language to a target language using the MarianMT model.
8. Chatbot Function:
    Main function which manages the conversation with the user.
    It initializes a dictionary (shared_info) to keep track of the user's provided information (name, email, phone).
    It includes a regular expression for detecting user greetings.
    The chatbot loops to continuously interact with the user.
    It performs the following actions:
    Checks for prompt injection and handles it.
    Checks if the user wants to update or correct data (name, email, phone).
    Conducts sentiment analysis on user input.
    Handles responses related to data privacy (email, phone, name).
    Responds to user greetings.
    Generates responses using the GPT-2 model with personalized information.
9. Generate Response Function:
    The generate_response function takes user input and shared information as input and generates a response using the GPT-2 model.
    It constructs responses based on the user's interaction and shared information.
