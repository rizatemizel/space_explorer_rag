import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Sidebar input for API keys
st.sidebar.title("API Configuration")

groq_api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# Check if the API key is provided
if not groq_api_key:
    st.warning("Please enter your GROQ API key to proceed.")
    st.stop()
else:
    st.success("Thank you for providing the API key! You can now start the chat.")

# Set up Huggingface embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the LLM using GROQ API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0)

# Define the prompt template
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

# Streamlit app layout and logic
st.markdown(
    """
    <h1 style='text-align: center; color: #4B9CD3; font-family: Arial, sans-serif; font-size: 24px;'>
    Assistant Tool for <br> 
    <i>European Cooperation for Space Standardization (ECSS) Publications</i>
    </h1>
    """, 
    unsafe_allow_html=True
)

st.image("space_image3.jpg")

# Stylized input prompt
user_prompt = st.text_input("🚀🚀🚀 How can I help you today?")

# Function to load the FAISS vector index from the saved file
def load_vector_embedding():
    vector_store_file = "faiss_vectors_index_MiniLM"  # Path to saved FAISS index

    try:
        if os.path.exists(vector_store_file):
            with st.spinner("Loading vector database from local storage..."):
                # Load the saved FAISS vectors from local storage
                st.session_state.vectors = FAISS.load_local(
                    vector_store_file, embeddings, allow_dangerous_deserialization=True
                )
            st.success("Vector store is ready. You can start chat.")
        else:
            st.error("FAISS vector index file not found! Ensure the file is in the correct location.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load vectors: {str(e)}")
        st.stop()

# Automatically load the FAISS index when the app starts
if "vectors" not in st.session_state:
    load_vector_embedding()

# If a user prompt is provided, execute the query
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("FAISS vector database not loaded.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 30})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Fetching the best results..."):
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                st.write(f"Response time: {time.process_time() - start}")

            st.write(response['answer'])

            # With a Streamlit expander to show document similarity search results
            with st.expander("Document similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        except Exception as e:
            st.error(f"Error retrieving document: {str(e)}")
