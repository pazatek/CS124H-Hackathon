import os
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
import gradio as gr

# Load environment variables from .env file
load_dotenv('.env')

# Get the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in .env file.")
print(f"API Key Loaded: {api_key}")

# Set the API key for the environment
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the language model
llm = ChatOpenAI(temperature=0.2)

# Path to the CSV file
csv_file_path = 'uiuc-gpa-dataset-five-years.csv'  # Ensure the file exists at this path

# Try to create the agent
agent_executor = None
try:
    agent_executor = create_csv_agent(llm, csv_file_path, verbose=True, allow_dangerous_code=True)
    print("Agent created successfully!")
except Exception as e:
    print(f"Failed to create agent: {e}")
    raise

# Ensure the agent is defined
if not agent_executor:
    raise RuntimeError("Agent creation failed.")

def process_query(query):
    try:
        response = agent_executor.invoke(query)
        # Extract only the 'output' from the response dictionary
        cleaned_response = response['output'] 
        return cleaned_response
    except Exception as e:
        return f"An error occurred while querying the agent: {e}"
    
# Create the Gradio interface with a custom layout
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column(scale=1):  # Output text area
            output_text = gr.Textbox(label="Alma's Response")
        with gr.Column(scale=1):  # Static image area
            output_image = gr.Image("AlmaBot.jpg", label="AlmaBot")  # Replace with your image file

    with gr.Row():
        # Input component
        input_text = gr.Textbox(
            lines=2,
            placeholder="Enter your question about courses or grades at UIUC here...",
            label="Your Question"
        )

    # Submit button to trigger the query
    btn = gr.Button("Submit")
    btn.click(fn=process_query, inputs=input_text, outputs=output_text)

# Launch the interface
iface.launch()