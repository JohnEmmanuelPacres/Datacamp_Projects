# pyright: ignore [reportMissingImports]
from langchain.chat_models import init_chat_model
# pyright: ignore [reportMissingImports]
from langchain_core.prompts import PromptTemplate
import os
# pyright: ignore [reportMissingImports]
from dotenv import load_dotenv
# pyright: ignore [reportMissingImports]
import gradio as gr

load_dotenv()
# change if different LLM
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")

prompt_temp_str = """
As a professional machine learning researcher, define the following **{concept}** 
Your task is to make sure the explanation is:
1. Clear and intuitive
2. Understandable and in-depth
3. Enable anyone to understand the topic clearly
4. Concise (in under 100 words)

Use the following information about me to personalize the explanation:
1. Computer engineering student
2. ML practitioner
3. Hardware and software enthusiast
4. Game development hobbyist

The personalization should be subtle and natural.

"""

prompt_template = PromptTemplate.from_template(prompt_temp_str)
model = init_chat_model("gemini-flash-latest", model_provider='google_genai')

def explain_machine_learning_concepts(concept):
    prompt = prompt_template.format(concept=concept)
    response = model.invoke(prompt)
    return(response.text)

demo = gr.Interface(
    fn = explain_machine_learning_concepts,
    inputs = [gr.Textbox(label="Enter ML concept:", lines=1)],
    outputs = [gr.Textbox(label="ML concept explanation: ", lines=5)],
    flagging_mode = "never",
    title = "Machine Learning Explainer",
    description = "Enter an ML concept and get a clear and concise explanation",
    theme = gr.themes.Soft()
)

demo.launch()