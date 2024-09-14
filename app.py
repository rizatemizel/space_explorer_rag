import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

# Custom CSS for header centering
st.markdown("""
    <style>
    h1 {
        text-align: center;
        color: #4B9CD3;
        font-family: Arial, sans-serif;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header (centered)
st.markdown(
    """
    <h1>
    Assistant Tool for <br> 
    <i>European Cooperation for Space Standardization (ECSS) Publications</i>
    </h1>
    """, 
    unsafe_allow_html=True
)

# Centering the image using st.columns
col1, col2, col3 = st.columns([1, 8, 1])  # The middle column will be larger
with col2:
    st.image("space_image3.jpg", use_column_width=True)  # Image is centered and scaled within the column

# Stylized input prompt
user_prompt = st.text_input("🚀🚀🚀 How can I help you today?")

# Sidebar: API Key submission form
with st.sidebar.form(key='api_form'):
    groq_api_key = st.text_input("Enter your GROQ API Key. You can have one from this link: https://console.groq.com/keys", type="password")
    submit_button = st.form_submit_button(label='Submit')

# Set up Huggingface embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to load the FAISS vector index from the saved file
@st.cache_resource(show_spinner=False)
def load_vector_embedding():
    vector_store_file = "faiss_vectors_index_MiniLM"  # Path to saved FAISS index
    try:
        if os.path.exists(vector_store_file):
            # Load the saved FAISS vectors from local storage
            vectors = FAISS.load_local(
                vector_store_file, embeddings, allow_dangerous_deserialization=True
            )
            return vectors
        else:
            st.error("FAISS vector index file not found! Ensure the file is in the correct location.")
            return None
    except Exception as e:
        st.error(f"Failed to load vectors: {str(e)}")
        return None

# Automatically load the FAISS index in the background
if "vectors" not in st.session_state:
    with st.spinner("Loading vector database in the background..."):
        st.session_state.vectors = load_vector_embedding()

# Check if the API key is provided
if groq_api_key:
    try:
        # Define the LLM using the provided API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0)

        # Test the API key with a dummy query to ensure it's valid
        test_query = "Test query to validate API key"
        
        # Make a small call to test the API key
        try:
            # Using the actual method available in ChatGroq class (assumed predict or generate here)
            # Substitute with actual available method
            llm_response = llm.predict(test_query)

            # If the call succeeds, show the success message
            st.sidebar.success("Thank you for providing the API key! You can now start the chat.", icon="✅")

        except Exception as test_error:
            # If the API key is invalid or the call fails
            st.sidebar.error("Invalid API key. Please enter a valid key.")
            st.stop()

        # Define the prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Ensure that your answers are as accurate as possible and always provide a reference. 
            Note that the context is derived from various documents, and these fragments do not originate from a single source. 
            Therefore, using a reference such as "Reference: 5.8.2.1" would not be useful as we cannot determine which document this subsection is from.
            When you mention tables or sub sections like 5.8.2.1, it would be very helpful to mention the standard name and number at the end.
            
            <context>
            {context}
            <context>
            Question: {input}
            """
        )

    except Exception as e:
        # If something goes wrong during the LLM setup or validation
        st.sidebar.error("Invalid API key. Please enter a valid key.")
        st.stop()
else:
    st.sidebar.warning("Please enter your GROQ API key to proceed.")

# If a user prompt is provided, execute the query
if user_prompt:
    if not groq_api_key:
        st.warning("Please enter your GROQ API key to submit a query.")
    elif "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("FAISS vector database not loaded.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(search_type="mmr", search_kwargs={'k': 15, 'fetch_k': 30})

            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Fetching the best results..."):
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                st.write(f"Response time: {time.process_time() - start}")

            st.write(response['answer'])

            # With a Streamlit expander to show document similarity search results
            with st.expander("Document Similarity Search (Retrieved Context)"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        except Exception as e:
            st.error(f"Error retrieving document: {str(e)}")
