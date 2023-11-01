from dotenv import load_dotenv
import os
from langchain import LLMChain, PromptTemplate
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

class App:

    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
            Classify the text bellow, delimited by three dashes (-), as having either a positive or negative sentiment.
            Answer with a single word: positive or negative. If you can't classify it, Say: Sorry I don't know.
            ---
            {message}
            ---

        """,
    )
    
    @classmethod
    def run(cls):
        st.title("Sentiment classifier")
        query = st.text_input("Enter something you want to classify as positive or negative")
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()

        if query:
            llm = OpenAI()
            chain = LLMChain(llm=llm, prompt=cls.prompt)
            with get_openai_callback() as cost:
                response = chain.run(query)
                print(cost)
            
            st.write(response)

if __name__ == '__main__':
    load_dotenv()
    App.run()