import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# ---------- Helper Functions ----------
def load_vector_embedding():
    """Load FAISS vector embedding from file."""
    vector_store_file = "faiss_vectors_index_MiniLM"
    if os.path.exists(vector_store_file):
        with st.spinner("Loading vector database from local storage..."):
            st.session_state.vectors = FAISS.load_local(
                vector_store_file, embeddings, allow_dangerous_deserialization=True
            )
        st.success("Vector store is ready...", icon="✅")
    else:
        st.error("FAISS vector index file not found! Ensure the file is in the correct location.")
        st.stop()

def query_llm(user_prompt):
    """Process user query and return response."""
    if "vectors" not in st.session_state:
        st.error("FAISS vector database not loaded.")
        return
    
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_type="mmr", search_kwargs={'k': 15, 'fetch_k': 30})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Fetching the best results..."):
            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            st.write(f"Response time: {time.process_time() - start}")

        # Display the response
        st.write(response['answer'])

        # Display retrieved documents in an expander
        with st.expander("Document Similarity Search (Retrieved Context)"):
            for doc in response['context']:
                st.write(doc.page_content)
                st.write('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    except Exception as e:
        st.error(f"Error retrieving document: {str(e)}")

# ---------- Frontend Design & Layout ----------
# Custom CSS for centered header and improved styling
st.markdown("""
    <style>
    h1 {
        text-align: center;
        color: #4B9CD3;
        font-family: Arial, sans-serif;
        font-size: 20px;
    }
    .stTextInput input {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #4B9CD3;
    }
    .stButton button {
        background-color: #4B9CD3;
        color: white;
        border-radius: 8px;
        border: 1px solid #4B9CD3;
        padding: 8px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Centered Header
st.markdown("""
    <h1>Assistant Tool for <br> 
    <i>European Cooperation for Space Standardization (ECSS) Publications</i>
    </h1>
""", unsafe_allow_html=True)

# Centering the image using columns
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.image("space_image3.jpg", use_column_width=True)

# Stylized input prompt
user_prompt = st.text_input("🚀🚀🚀 How can I assist you today?")

# Sidebar API Key submission
with st.sidebar.form(key='api_form'):
    groq_api_key = st.text_input("Enter your GROQ API Key. You can get one from [here](https://console.groq.com/keys)", type="password")
    submit_button = st.form_submit_button(label='Submit')

# Handling API Key validation and status
if not groq_api_key:
    st.sidebar.warning("Please enter your GROQ API key to proceed.")
    st.stop()  # Stop the app until the user provides the API key
else:
    st.sidebar.success("API key submitted successfully!", icon="✅")

    # Now initialize everything that requires the API key after it's provided
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the LLM with the submitted API key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0)

    # Defining the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Ensure that your answers are as accurate as possible and always provide a reference. 
        Note that the context is derived from various documents, and these fragments do not originate from a single source. 
        Therefore, using a reference such as "Reference: 5.8.2.1" would not be useful as we cannot determine which document this subsection is from.    
        <context>
        {context}
        <context>
        Question: {input}
        """
    )

    # Load FAISS vectors only after API key is submitted
    if "vectors" not in st.session_state:
        load_vector_embedding()

# Process user input
if user_prompt:
    if not groq_api_key:
        st.warning("Oops! It seems like you forgot to enter your GROQ API key. If you don't have one, getting it is easy! Please visit [this link](https://console.groq.com/keys) to generate an API key.")
    else:
        query_llm(user_prompt)
