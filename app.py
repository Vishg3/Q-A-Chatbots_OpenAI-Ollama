import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OpenAI"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user's queries."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,engine,temperature,max_tokens):
    llm=ChatOpenAI(model=engine,temperature=temperature,max_tokens=max_tokens,openai_api_key=api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

st.title("Q&A Chatbot with OpenAI")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your OpenAI api key",type="password")
engine=st.sidebar.selectbox("Choose your OpenAI model",["gpt-4o","gpt-4-turbo","gpt-4"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max tokens",min_value=50,max_value=300,value=150)
st.write("Ask me anything")
user_input=st.text_input("Your input:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter your OpenAI api key in the sidebar")
else:
    st.write("Please provide your query")