from llm_apis import ValidLLMs, get_llm_api, HUITOpenAI

# Initialize the chat model
chat_model = HUITOpenAI('gpt-4o')

# Define the survey questions
survey_questions = [
    "The explanation helps me understand how the LLM rule-based policy works. Please answer Yes or No.",
    "The explanation of how the LLM rule-based policy works is satisfying. Please answer Yes or No.",
    "The explanation of the LLM rule-based policy is sufficiently detailed. Please answer Yes or No.",
    "The explanation of how the LLM rule-based policy works is sufficiently complete. Please answer yes or No."
]

# Initialize the conversation with a system message
question_prompt = """
Task: You are assisting doctors from a hospital in making optimized decisions about which patient should receive a vital sign monitor device. It is critical to prioritize patients based on their needs.

Possible actions: Choose the id of the device that will be reallocated to the new incoming patient. Your answer should be a single integer `i` from 0 to 4.
- Always choose a free device if available.
- If no free device is available, then choose device `i` whose current patient is at least risk or would benefit less from wearing the device.

Current state of the decision problem:
Number of devices: 5
Number of free devices: none

Device 0: Device is currently assigned to a patient with the following description:
- Timesteps wearing the device: 6
- Pulse rate: Last value: 82.90, Mean: 87.19, Standard deviation/volatility: 4.07
- Respiratory rate: Last value: 26.09, Mean: 26.15, Standard deviation/volatility: 0.75
- SPO2: Last value: 96.76, Mean: 97.13, Standard deviation/volatility: 1.96

Device 1: Device is currently assigned to a patient with the following description:
- Timesteps wearing the device: 3
- Pulse rate: Last value: 94.33, Mean: 96.52, Standard deviation/volatility: 4.90
- Respiratory rate: Last value: 33.55, Mean: 22.66, Standard deviation/volatility: 7.40
- SPO2: Last value: 97.17, Mean: 97.88, Standard deviation/volatility: 0.43

Device 2: Device is currently assigned to a patient with the following description:
- Timesteps wearing the device: 2
- Pulse rate: Last value: 69.85, Mean: 66.99, Standard deviation/volatility: 2.60
- Respiratory rate: Last value: 22.18, Mean: 21.76, Standard deviation/volatility: 4.35
- SPO2: Last value: 96.84, Mean: 97.79, Standard deviation/volatility: 0.56

Device 3: Device is currently assigned to a patient with the following description:
- Timesteps wearing the device: 4
- Pulse rate: Last value: 68.67, Mean: 70.78, Standard deviation/volatility: 2.97
- Respiratory rate: Last value: 24.07, Mean: 21.60, Standard deviation/volatility: 2.36
- SPO2: Last value: 99.89, Mean: 96.11, Standard deviation/volatility: 2.08
"""

messages = [
    {'role': 'system', 'content': question_prompt}
]

# Store GPT responses
responses = {}

# Iterate over the survey questions
for i, question in enumerate(survey_questions, start=1):
    # Add the user question to the messages
    messages.append({'role': 'user', 'content': question})
    
    # Get the response using the invoke method
    response = chat_model.invoke(messages)
    
    # Extract the assistant's reply
    answer = response.content  # Assuming `response.content` holds the reply text
    
    # Store the response
    responses[f"Question {i}"] = answer
    
    # Add GPT's response to the messages for multi-turn context
    messages.append({'role': 'assistant', 'content': answer})

# Save the responses to a text file
with open('survey_responses.txt', 'w') as file:
    # First, write the question prompt
    file.write("=== Question Prompt ===\n")
    file.write(question_prompt + "\n\n")

    # Then, write the survey responses
    file.write("=== Survey Responses ===\n")
    for question, answer in responses.items():
        file.write(f"{question}: {answer}\n")

# Print a confirmation message
print("Survey responses saved to 'survey_responses.txt'")
