from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

load_dotenv()
st.title("Fee Management System")
df = pd.read_excel("Fee Management.xlsx")
st.dataframe(df.head())
model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id = "MiniMaxAI/MiniMax-M2.7",
        task = "text-generation"
    )
)
question = st.text_input("Ask your question here")
if st.button("Get Answer"):
    if question:
        prompt = f""""
        Yor are a helpful assistant. Answer the user's question ONLY from the given 
        dataset below. If the answer is not available in the dataset, reply 
        exactly: Sorry, I don't know the answer about this.
        
        
        Dataset:{df.to_string(index=False)}
        Question: {question}"""
        
        response = model.invoke(prompt)
        st.write(response.content)
    else:
        st.write("Please enter a question.")




