import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from rag_chain import chain   # your RAG chain

load_dotenv()

st.title("RAG Document QnA System")

uploaded_file = st.file_uploader("Upload a document", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("Document uploaded successfully!")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# When user sends a new prompt
if prompt := st.chat_input("Ask a question about the document"):

    # Store user msg
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call RAG chain
    with st.chat_message("assistant"):
        response = chain.invoke({"question": prompt})
        st.markdown(response)

        # Store AI msg
        ai_msg = AIMessage(content=response)
        st.session_state.messages.append(ai_msg)
