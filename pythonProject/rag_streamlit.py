import streamlit as st
import tiktoken
import os
from logger import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

def main():
    st.set_page_config(
        page_title="JieunChat",
        page_icon=":computer:")

    st.title("_Question Any Data :mag: :blue[QA :]_")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        # 파일 업로드와 URL 입력 중 하나를 선택하게 하는 체크박스
        input_option = st.radio("문서를 업로드하거나 URL을 입력하세요:", ("파일 업로드", "URL 입력"))

        if input_option == "파일 업로드":
            uploaded_files = st.file_uploader("파일을 업로드 하세요", type=['pdf', 'docx'], accept_multiple_files=True)
        else:
            url_input = st.text_input("웹 페이지 URL을 입력하세요")

        openai_api_key = os.getenv(
            "OPENAI_API_KEY")  # st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 대화 기록 및 메시지 초기화
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 문서를 등록하거나 URL을 입력하고 궁금한 것을 물어보세요!"}]
        st.session_state.chat_history = None

        # 파일 업로드나 URL 입력 중 하나에 따라 처리
        if input_option == "파일 업로드":
            if not uploaded_files:
                st.warning("파일을 업로드해주세요.")
                st.stop()
            files_text = get_text(uploaded_files)
        elif input_option == "URL 입력":
            if not url_input:
                st.warning("URL을 입력해주세요.")
                st.stop()
            files_text = get_text_from_url(url_input)

            # URL에서 문서가 제대로 로드되었는지 확인
            if not files_text or len(files_text) == 0:
                st.warning("URL에서 데이터를 불러오지 못했습니다. URL을 확인하세요.")
                st.stop()

        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 문서를 등록하거나 URL을 입력하고 궁금한 것을 물어보세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    if len(source_documents) > 0:
                        st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    if len(source_documents) > 1:
                        st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []
    st.session_state['messages'] = [{"role": "assistant", "content": "질문을 입력해주세요."}]
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_from_url(url):
    st.warning(url)
    loader = WebBaseLoader(url)
    documents = loader.load()

    # documents가 비어 있거나 유효한 데이터를 불러오지 못했을 경우 처리
    if not documents:
        st.warning("URL에서 데이터를 불러오지 못했습니다. URL을 확인하세요.")
        st.stop()

    # 불러온 데이터를 출력하여 확인
    st.write(f"Loaded {len(documents)} documents from the URL")
    st.session_state['messages'] = [{"role": "assistant", "content": "질문을 입력해주세요."}]
    return documents


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        #model_name="jhgan/ko-sroberta-multitask",
        model_name="nlpai-lab/KoE5",
        model_kwargs={'device': 'cpu'}, #cuda
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o-mini', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain


if __name__ == '__main__':
    main()
