import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document 

# Existing function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Existing function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Existing function to create a vector store from text chunks
def get_vector_store(text_chunks, google_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Existing function to set up a conversational chain for Q&A
def get_conversational_chain(google_api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Existing function to handle user input for Q&A
def user_input(user_question, google_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(google_api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# New function for quiz generation
def generate_quiz(text, google_api_key):
    prompt_template = """
    From the provided content, generate 5 multiple-choice questions with 4 options and indicate the correct answer.\n\n
    Content:\n {context}\n

    Quiz:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    
    # Updated to pass documents properly
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    # Create Document objects from text
    docs = [Document(page_content=text)]  # Convert the text into a Document object
    
    # Generate quiz
    quiz_response = chain({"input_documents": docs}, return_only_outputs=True)
    return quiz_response["output_text"]


# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF with Quiz")
    st.header("Chat with PDF and Generate Quiz using Gemini💁")
    
    # Input for Google API key
    google_api_key = st.text_input("Enter your Google API Key", type="password")
    
    if not google_api_key:
        st.warning("Please enter your Google API Key")
        return
    
    genai.configure(api_key=google_api_key)
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question, google_api_key)
    
    with st.sidebar:
        st.title("Menu:")
        
        # PDF Upload Section
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, google_api_key)
                st.success("Done")
        
        # Quiz Generation Section
        if st.button("Generate Quiz from PDF"):
            with st.spinner("Generating Quiz..."):
                raw_text = get_pdf_text(pdf_docs)
                quiz = generate_quiz(raw_text, google_api_key)
                st.subheader("Generated Quiz:")
                st.write(quiz)

if __name__ == "__main__":
    main()
