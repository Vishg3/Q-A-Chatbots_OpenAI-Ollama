import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with Ollama"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user's queries."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature):
    llm=Ollama(model=engine,temperature=temperature)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

st.title("Q&A Chatbot with Ollama")
st.sidebar.title("Settings")
engine=st.sidebar.selectbox("Choose your Ollama model",["llama3:latest"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
st.write("Ask me anything")
user_input=st.text_input("Your input:")

if user_input:
    response=generate_response(user_input,engine,temperature)
    st.write(response)
else:
    st.write("Please provide your query")