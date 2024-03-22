import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter      # split the text in the document
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader   # To load the pdf
from langchain.chains.summarize import load_summarize_chain

#  tokenizing input text into a format that the T5 model can understand.
#  generate text based on a given condition
from transformers import T5Tokenizer,T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64


checkpoint = "LaMini_Flan"
# T5Tokenizer is a class from HuggingFace designed for tokenizing text data
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint,device_map ='auto',torch_dtype = torch.float32)
# device_map = helps to  automatically determine the appropriate device based on the available hardware
# torch_dtype = specifies the data type for the model parameters

def file_preprocessing(file):
    loader = PyPDFLoader(file)              # in order to load the PDF 
    pages = loader.load_and_split()         # split into indivisual pages
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)        # split the texts into chunks and each chuck will have size 200 
    # chunk_overlap is used so that no imformation is lost

    texts = text_splitter.split_documents(pages)    # split thee content of the pages into individual texts 
    final_texts = ""
    for text in texts:
        print(text)
        final_texts  = final_texts +  text.page_content
    return final_texts


# Define LLM Pipline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer, 
        max_length = 500,
        min_length = 50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)

    result = result[0]['summary_text']

    return result


@st.cache_data

# Function to display PDF 
def displayPDF(file):
    # opening  file
    with open(file,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8') 

    pdf_display = F'<iframe src= "data:application/pdf;base64,{base64_pdf}" width="100%" height= "600" type="application/pdf"></iframe'

    # Displaying file
    st.markdown(pdf_display,unsafe_allow_html = True)


# Streamlit

st.set_page_config(layout='wide',page_title = "Summarization")


def main():
    st.title("Document Summarization")

    uploaded_file = st.file_uploader("Upload your PDF",type= ['pdf'])
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1,col2 = st.columns(2)
            filepath = "Data/" + uploaded_file.name
            with open(filepath, 'wb') as temp:
                temp.write(uploaded_file.read())
            
            with col1:
                st.info("Uploaded PDF file")
                pdf_viewer =  displayPDF(filepath)
               
            with col2:  
                st.info("Summaization is below")
                summary = llm_pipeline(filepath)
                st.success(summary)


if __name__ == '__main__':
    main()
