# Advanced Conversational Chatbot 

**Requirements**

1. Python 3.8
2. Libraries: Transformers, re, logging, MySQL, Flask

**Usage**

1. Clone the repository or download the Python script.
2. Ensure that the required libraries are installed.
3. Run the script using the following command: `python main.py`
4. Flask REST API endpoints available in `app.py`

**Functionality**

1. **Prompt Injection Detection:** The chatbot checks for potential prompt injections in user input and logs any detected malicious input.

2. **Data Validation:** The script validates user-provided data such as names, emails, and phone numbers for their correctness and format.

3. **Sentiment Analysis:** It performs sentiment analysis on user inputs to determine the sentiment of the messages.

4. **Language Translation:** The chatbot can translate text from one language to another using the MarianMT model.

5. **Dynamic Blacklisting:** It prevents harmful prompts by blacklisting specific phrases.

6. **Named Entity Recognition:** The script extracts names, emails, and phone numbers from user inputs. 


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

**Output**

https://github.com/K-R-U-T-I/advanced_conversational_chatbot_CN/assets/38699938/fcd25942-3288-4003-ba0f-276ba808e762

![op5](https://github.com/K-R-U-T-I/advanced_conversational_chatbot_CN/assets/38699938/a9c10ce5-db77-47c1-ba1c-b2b78c4644f5)
![op4](https://github.com/K-R-U-T-I/advanced_conversational_chatbot_CN/assets/38699938/44eb5e81-59a7-4c20-ba66-e6642b0cbbb3)
![op3](https://github.com/K-R-U-T-I/advanced_conversational_chatbot_CN/assets/38699938/00019421-74fe-4ce4-ad33-2cd00cbc9c25)
![op2](https://github.com/K-R-U-T-I/advanced_conversational_chatbot_CN/assets/38699938/4e45284b-36c4-4e18-83d6-744bbd9a88ab)
![op1](https://github.com/K-R-U-T-I/advanced_conversational_chatbot_CN/assets/38699938/c6d817b4-7e96-4d8f-a011-abb651fa7e51)




