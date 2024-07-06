import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

load_dotenv()

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to initialize conversational AI chain
def get_conversational_chain(vector_store):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate response
def handle_user_input(user_question, vector_store, conversation_chain):
    docs = vector_store.similarity_search(user_question)
    response = conversation_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.markdown(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    st.header("Your Physics Buddy :books:")
    user_question = st.text_input("Ask a question about your documents:")

    # Handle user input
    if user_question:
        if st.session_state.vector_store is not None and st.session_state.conversation_chain is not None:
            response_text = handle_user_input(user_question, st.session_state.vector_store, st.session_state.conversation_chain)
            st.markdown(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            st.markdown(bot_template.replace("{{MSG}}", response_text), unsafe_allow_html=True)

    # Sidebar for uploading PDFs and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
                st.session_state.conversation_chain = get_conversational_chain(st.session_state.vector_store)
                st.success("Done")

if __name__ == "__main__":
    main()
