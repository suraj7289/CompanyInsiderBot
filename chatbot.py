import streamlit as st
from langchain.llms import AzureOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from factory import api_key,api_base,api_version,deployment_name, model_name,embed_deployment_name, embed_model_name
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import re
import pickle
from tiktoken import get_encoding
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = api_version
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key


def get_vectorstore(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
        else:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            embeddings = OpenAIEmbeddings(deployment=embed_deployment_name, model=embed_model_name, chunk_size=16)
            vectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)
    return vectorStore


with st.sidebar:
    st.title('CompanyInsiderBot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')

st.header("Find your answers from Annual Report 💬")

# upload a PDF file
pdf_docs = st.file_uploader("Upload your report", type='pdf', accept_multiple_files=True)
# get vectorestore for pdfs
if pdf_docs:
    vectorstore = get_vectorstore(pdf_docs)

query = st.text_input("Ask your question from uploaded Annual Report")
if query:
    docs = vectorstore.similarity_search(query = query)
    llm1 = AzureOpenAI(deployment_name=deployment_name, model_name=model_name, temperature= 0, top_p = 1, max_tokens = 1024)
    llm2 = AzureOpenAI(deployment_name=deployment_name, model_name=model_name, temperature= 0, top_p = 1, max_tokens=512)

    prompt1 = ''' "Instruction" : You are a chatbot. I am feeding you document text and a question. Based on my query, Goal is to extract the all relevant details from the given text. Don't generate hallucinating answers, response should be extracted from the text and should retain the factual data and details. Please keep the response concise and avoid duplicate information as much as possible. Avoid junk information in response. Give answer with in max_token limit of llm. "Question" : ''' + query
    chain1 = load_qa_chain(llm=llm1, chain_type="stuff")
    chain2 = load_qa_chain(llm=llm2, chain_type="stuff")
    with get_openai_callback() as cb:
        llmresponse1 = chain1.run(input_documents = docs, question=prompt1)
        #print(cb)
    llmresponse1 = llmresponse1.rstrip('<|im_end|>')

    prompt2 = '''"Instruction": One LLM has generated a detailed summary of my previous question. I am feeding you output of first LLM.
    Use the given summary text as context to find the answer to the provided question. Stick to the question and provide a relevant answer, avoiding any unrelated responses. If you don't find any answer, just respond - "Sorry, I did not get any answer for your question from given documents. Please try with the other questions."
    1. If the summary text contains only natural language text, provide a concise summary in response. Don't generate any json data and avoid any redundant information in response. Try to make the response more readable by adding sections, sub-sections, and bullet points wherever necessary.
    2. If the summary text contains any details to generate a tabular data, create a json response in below format - 
    {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
    If query says to create a chart, follow this format :
    {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
    "Summary Text" : ''' + llmresponse1 + " Question: " + query
    with get_openai_callback() as cb:
        llmresponse2 = chain2.run(input_documents = docs, question=prompt2)
        #print(cb)
    llmresponse2 = llmresponse2.rstrip('<|im_end|>')
    json_pattern = r"\{.+?\}"
    json_strings = re.findall(json_pattern, llmresponse2)
    if len(json_strings)>0:
        jsonprompt = "Write a brief descriptive overview of above json file. Don't add technical details of file, make it explainable to business people. json : " + json_strings[0]
        jsonsummary = llm2(jsonprompt)
        jsonsummary = jsonsummary.rstrip('<|im_end|>')
        st.write(jsonsummary)
        st.write(llmresponse2)
    else:
        st.write(llmresponse2)
