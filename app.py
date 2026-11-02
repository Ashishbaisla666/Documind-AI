import streamlit as st

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------- CONFIGURATION ---------------- #

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


# ---------------- PDF EXTRACTION ---------------- #

def get_pdf_text(pdf_docs):

    documents = []

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page_number, page in enumerate(pdf_reader.pages):

            text = page.extract_text() or ""

            if text.strip():

                documents.append({
                    "text": text,
                    "source": pdf.name,
                    "page": page_number + 1
                })

    return documents


# ---------------- TEXT CHUNKING ---------------- #

def get_text_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    from langchain_core.documents import Document

    langchain_documents = []

    for item in documents:

        langchain_documents.append(
            Document(
                page_content=item["text"],
                metadata={
                    "source": item["source"],
                    "page": item["page"]
                }
            )
        )

    chunks = splitter.split_documents(langchain_documents)

    return chunks


# ---------------- VECTOR DATABASE ---------------- #

def get_vectorstore(text_chunks):

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# ---------------- AI CONVERSATION ---------------- #

def get_ai_response(question, vectorstore, chat_history):

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    # Retrieve relevant document chunks

    relevant_docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = ""

    for doc in relevant_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context += f"""
        Source: {source}
        Page: {page}

        Content:
        {doc.page_content}

        ------------------------
        """

    # Include previous conversation

    history = ""

    for user_msg, bot_msg in chat_history[-5:]:

        history += f"""
        User: {user_msg}
        Assistant: {bot_msg}
        """

    prompt = f"""
    You are DocuMind AI, an intelligent document
    question-answering assistant.

    Answer the user's question using the provided
    document context.

    Instructions:

    1. Give accurate and clear answers.
    2. Use the uploaded documents as your primary source.
    3. If the answer is not available in the documents,
       clearly say that the information was not found.
    4. Do not invent facts.
    5. Provide structured answers where appropriate.

    Previous Conversation:
    {history}

    Document Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content, relevant_docs


# ---------------- DISPLAY CHAT ---------------- #

def display_chat():

    for message in st.session_state.chat_history:

        with st.chat_message("user"):

            st.write(message["question"])

        with st.chat_message("assistant"):

            st.write(message["answer"])

            with st.expander("View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['name']} | "
                        f"Page {source['page']}"
                    )


# ---------------- MAIN APPLICATION ---------------- #

def main():

    st.title("📚 DocuMind AI")

    st.caption(
        "Intelligent Multi-Document Question Answering System"
    )

    st.markdown(
        "Upload multiple PDF documents and ask questions "
        "using locally running AI models."
    )

    # Initialize session state

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar

    with st.sidebar:

        st.header("📂 Document Management")

        pdf_docs = st.file_uploader(
            "Upload your PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button(
            "Process Documents",
            use_container_width=True
        ):

            if not pdf_docs:

                st.warning("Please upload at least one PDF.")

            else:

                try:

                    with st.spinner(
                        "Reading and processing documents..."
                    ):

                        raw_documents = get_pdf_text(pdf_docs)

                        if not raw_documents:

                            st.error(
                                "No readable text found in PDFs. "
                                "Scanned PDFs may require OCR."
                            )

                        else:

                            text_chunks = get_text_chunks(
                                raw_documents
                            )

                            vectorstore = get_vectorstore(
                                text_chunks
                            )

                            st.session_state.vectorstore = vectorstore

                            st.session_state.chat_history = []

                            st.success(
                                f"Successfully processed "
                                f"{len(pdf_docs)} PDF(s)!"
                            )

                            st.info(
                                f"Created {len(text_chunks)} "
                                "document chunks."
                            )

                except Exception as e:

                    st.error(f"Processing error: {str(e)}")

        st.divider()

        if st.session_state.vectorstore:

            st.success("Documents ready for questions.")

        if st.button(
            "Clear Conversation",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()

    # Main chat interface

    display_chat()

    user_question = st.chat_input(
        "Ask something about your documents..."
    )

    if user_question:

        if st.session_state.vectorstore is None:

            st.warning(
                "Please upload and process your PDFs first."
            )

        else:

            with st.chat_message("user"):

                st.write(user_question)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:

                        answer, sources = get_ai_response(
                            user_question,
                            st.session_state.vectorstore,
                            st.session_state.chat_history
                        )

                        st.write(answer)

                        source_list = []

                        for doc in sources:

                            source_list.append({
                                "name": doc.metadata.get(
                                    "source", "Unknown"
                                ),
                                "page": doc.metadata.get(
                                    "page", "Unknown"
                                )
                            })

                        with st.expander("View Sources"):

                            for source in source_list:

                                st.write(
                                    f"📄 {source['name']} | "
                                    f"Page {source['page']}"
                                )

                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": answer,
                            "sources": source_list
                        })

                    except Exception as e:

                        st.error(f"AI Error: {str(e)}")


# ---------------- APPLICATION ENTRY ---------------- #

if __name__ == "__main__":

    main()