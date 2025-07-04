# import streamlit as st




# st.set_page_config(page_title="Talk with PDF",
#                        page_icon="ui/assets/icon.png")

# st.title("Talk with PDF")


# with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your Data here  in PDF format and click on 'Process'", accept_multiple_files=True, type=['pdf'])
   


# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# st.file_uploader("Upload a CSV")

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
    
# if prompt := st.chat_input("What is up?"):

    
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    


# response = f"FireGPT: {prompt}"
# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     st.markdown(response)
# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": response})



import streamlit as st

# Custom CSS to move the sidebar to the right
st.markdown(
    """
    <style>
    .css-1lcbmhc {
        position: fixed;
        right: 0;
        top: 0;
        height: 100vh;
        width: 25rem;
        z-index: 1000;
        background-color: white;
        box-shadow: -2px 0 5px rgba(0,0,0,0.1);
    }
    .main .block-container {
        padding-left: 2rem;
        padding-right: 30rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="FireGPT", page_icon="ui/assets/icon.png")

# Add an image before the title
st.image("ui/assets/icon.png", width=300)

# Title
st.title("FireGPT")

# Right sidebar
with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your Data here in PDF format and click on 'Process'",
        accept_multiple_files=True,
        type=['pdf']
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.file_uploader("Upload Pictures")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"FireGPT: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
