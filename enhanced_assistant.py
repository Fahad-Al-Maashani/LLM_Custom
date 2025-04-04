# enhanced_assistant.py

import torch
import torch.nn.functional as F
from gpt_model import GPT  # Assume this is your GPT model file

# Dummy functions for rule-based responses
def handle_html_request():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Counter Website</title>
</head>
<body>
    <h1>Counter</h1>
    <p id="counter">0</p>
    <button onclick="document.getElementById('counter').innerText = parseInt(document.getElementById('counter').innerText) + 1;">Increase</button>
</body>
</html>"""

def handle_math_request(prompt):
    try:
        # Very naive math evaluation from prompt
        # Extracting math expression by removing non-digit/operator characters (improve as needed)
        expression = ''.join(ch for ch in prompt if ch in "0123456789+-*/(). ")
        result = eval(expression)
        return f"The result of {expression.strip()} is {result}."
    except Exception as e:
        return f"Error evaluating math expression: {e}"

def handle_code_request():
    return """# Sample Python code
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()"""

def handle_email_request():
    return """Subject: Meeting Request

Hi [Recipient Name],

I hope you're doing well. I would like to schedule a meeting to discuss our upcoming project.

Best regards,
[Your Name]"""

def generate_model_response(model, prompt_ids, max_new_tokens=50, temperature=1.0):
    model.eval()
    input_ids = torch.tensor([prompt_ids])
    for _ in range(max_new_tokens):
        if input_ids.shape[1] > model.max_seq_length:
            input_ids = input_ids[:, -model.max_seq_length:]
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids[0].tolist()

def simple_tokenizer(text, vocab_size=10000):
    tokens = text.split()
    token_ids = [abs(hash(token)) % vocab_size for token in tokens]
    return token_ids

def simple_detokenizer(token_ids):
    return " ".join([str(token) for token in token_ids])

def process_request(user_input, model, vocab_size):
    input_lower = user_input.lower()
    if "html" in input_lower or "website" in input_lower or "counter" in input_lower:
        return handle_html_request()
    elif "solve" in input_lower and "math" in input_lower:
        return handle_math_request(user_input)
    elif "code" in input_lower:
        return handle_code_request()
    elif "email" in input_lower:
        return handle_email_request()
    else:
        # Fall back to model generation
        prompt_ids = simple_tokenizer(user_input, vocab_size)
        generated_ids = generate_model_response(model, prompt_ids)
        return simple_detokenizer(generated_ids)

def chat():
    # Hyperparameters (for demonstration)
    vocab_size = 10000
    emb_size = 256
    num_layers = 4
    n_heads = 8
    max_seq_length = 128
    model = GPT(vocab_size, emb_size, num_layers, n_heads, max_seq_length)
    
    print("Welcome to the Advanced Multi-Purpose LLM Assistant!")
    print("You can ask for HTML generation, math problem solving, code generation, email composition, etc.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break
        # Decide response based on request
        response = process_request(user_input, model, vocab_size)
        print("Assistant:", response, "\n")

if __name__ == "__main__":
    chat()
