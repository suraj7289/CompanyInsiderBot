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
from tiktoken import get_encoding
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = api_version
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key


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
pdf = st.file_uploader("Upload your report", type='pdf')
if pdf is not None:
    pdf_reader = PdfReader(pdf)

    text = ""
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    )
    pdfTexts = text_splitter.split_text(raw_text)

    ## embeddings
    store_name = pdf.name[:-4]
    #st.write(f'{store_name}')
    ##st.write(len(pdfTexts))

    embeddings = OpenAIEmbeddings(deployment=embed_deployment_name,model = embed_model_name, chunk_size=16)
    pdfDocSearch = FAISS.from_texts(pdfTexts, embedding = embeddings)

    query = st.text_input("Ask your question from uploaded Annual Report")
    if query:
        docs = pdfDocSearch.similarity_search(query = query)
        llm1 = AzureOpenAI(deployment_name=deployment_name, model_name=model_name, temperature= 0, top_p = 1, max_tokens = 1024)
        llm2 = AzureOpenAI(deployment_name=deployment_name, model_name=model_name, temperature= 0, top_p = 1, max_tokens=512)

        prompt1 = "You are a chatbot. I am feeding you document text and a question. Based on my query, Goal is to extract the all relevant details from the given text. Don't generate hallucinating answers, response should be extracted from the text and should retain the factual data and details. Please keep the response concise and avoid duplicate information as much as possible. Avoid junk information in response. Give answer with in max_token limit of llm. Question : " + query
        chain1 = load_qa_chain(llm=llm1, chain_type="stuff")
        chain2 = load_qa_chain(llm=llm2, chain_type="stuff")
        with get_openai_callback() as cb:
            llmresponse1 = chain1.run(input_documents = docs, question=prompt1)
            #print(cb)
        llmresponse1 = llmresponse1.rstrip('<|im_end|>')

        prompt2 = '''One LLM has generated a detailed summary of my previous question. I am feeding you output of first LLM. Follow all instructions in sequence.
        1. Use given Summary Text as context and Find answer to my Question. Stick to the question and respond with answer of this question. Do not give any hallucinating response.
        2. If you find details in the summary text that is sufficient to generate a tabular data, create json from it. Json should have all necessary details to build a dataframe. If you generate json, no additional text should be included in response other than json.
        3. If you don't see any details that can help me build a tabular data, summarize the text and respond, don't give any json in this case. Keep summary concise, avoid redundant information in response.
        4. Response should not exceed 10 sentences. Don't generate other questions and references.
        5. If you are giving json response, make sure no additional text included in response.
        6. If you are giving summary text in response, make it more readable by adding section, sub-section and bullet-points wherever required.
        Summary Text : ''' + llmresponse1 + " Question: " + query
        with get_openai_callback() as cb:
            llmresponse2 = chain2.run(input_documents = docs, question=prompt2)
            #print(cb)
        llmresponse2 = llmresponse2.rstrip('<|im_end|>')
        llmresponse2 = llmresponse2.rstrip('{}')
        pattern = r'\{.+?\}'
        matches = re.findall(pattern, llmresponse2, re.DOTALL)
        if matches:
            botresponse = matches[0]
            st.write(botresponse)
        else:
            st.write(llmresponse2)
