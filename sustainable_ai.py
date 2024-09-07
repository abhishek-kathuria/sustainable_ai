# Define project information
# PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
BUCKET_NAME = 'sustainable_ai_hackathon'
PROJECT_ID = "playground-abihishek-bits"  # @param {type:"string"}
import vertexai
from  vertexai import generative_models
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

multimodal_model = GenerativeModel(
    "gemini-1.5-flash-001",
    
    system_instruction=["""
    You are an AI trained in sustainable technologies, specializing in suggesting the most CO2-efficient Google Cloud Platform (GCP) technical stack according to the user's specific use case. 
    In responding, consider the following steps:
    - Analyze Requirements: Based on the provided use case, analyze the computational intensity, storage needs, and potential scalability. This analysis will help in selecting the most suitable and eco-friendly GCP services.
    - Suggest GCP Services: Recommend a set of GCP services that align with the CO2 efficiency goals. Include options for computing services, storage solutions, and any relevant management tools.
    - Explain Your Choices: For each suggested service, explain why it is considered CO2-efficient in the context of the user needs. Discuss any trade-offs and suggest best practices for optimizing resource usage.
    Finally, answer the following question based on the context documentation provided. Always support your answer with clear, step-by-step reasoning that explains your choices. 

"""]
    )



# Construct the prompt for each input and its corresponding output JSON
prompt = []

context_document1 = Part.from_uri(
    mime_type="application/pdf",
    uri="gs://sustainable_ai_hackathon/documents/documents_accelerating-climate-action-ai.pdf")

context_document2 = Part.from_uri(
    mime_type="application/pdf",
    uri="sustainable_ai_hackathon/documents/documents_alphabet-2023-cdp-climate-change-response.pdf")

prompt = ["Here are the context documents:", "Document 1:",context_document1,  "User Question:"]

print(prompt)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.5,
    "top_p": 0.9,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
# LLM Model
def generate_response(prompt):
    responses = multimodal_model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False    
    )
    return responses

user_query="I have 100s of pdf documents. I want to design a PDF to JSON extractor using Gemini. Suggest me a GCP solution stack to build this out"

prompt.append(user_query)

print(prompt)

response = generate_response(prompt)

from IPython.display import display, Markdown

# Display the text as markdown
display(Markdown(response.text))