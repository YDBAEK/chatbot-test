import os
import tempfile
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory


# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ✅ SerpAPI 검색 툴 정의 (제목 + 링크 + 출처 + 스니펫)
def search_web():
    search = SerpAPIWrapper()

    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet", "")
            if link:
                formatted.append(f"- {title} ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."

    return Tool(
        name="web_search",
        func=run_with_source,
        description=(
            "실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. "
            "결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
        ),
    )


# ✅ PDF 업로드 → 벡터DB → 검색 툴 생성 (+메타데이터 보강: file_name, page)
def load_pdf_files(uploaded_files):
    all_documents = []

    # 원본 파일명 보존을 위해 메타데이터 삽입
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        for d in documents:
            d.metadata["file_name"] = uploaded_file.name  # 원본 업로드 파일명
            # PyPDFLoader는 보통 'page' 메타데이터를 포함함 (0-based). 없으면 None.

        all_documents.extend(documents)

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    # 파일별 그룹핑 (요약에 사용)
    grouped_by_file = {}
    for d in split_docs:
        fname = d.metadata.get("file_name", "unknown.pdf")
        grouped_by_file.setdefault(fname, []).append(d)

    # 벡터DB 생성
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    # ✅ 커스텀 PDF 검색 툴: 스니펫 + (파일명, 페이지) 함께 반환
    def run_pdf_search(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "PDF에서 관련 정보를 찾지 못했습니다."

        lines = []
        for i, d in enumerate(docs[:5]):
            file_name = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", ""))
            page_meta = d.metadata.get("page")
            # PyPDFLoader는 0-based 페이지. 사람이 보기 좋게 +1
            page_disp = page_meta + 1 if isinstance(page_meta, int) else page_meta
            snippet = d.page_content.strip().replace("\n", " ")
            lines.append(
                f"{i+1}. {snippet}\n   (출처: {file_name}, p.{page_disp})"
            )
        return "\n".join(lines)

    retriever_tool = Tool(
        name="pdf_search",
        func=run_pdf_search,
        description=(
            "업로드된 PDF 문서에서 정보를 검색합니다. "
            "반환 형식은 스니펫과 함께 (출처: 파일명, p.페이지) 정보를 포함합니다."
        ),
    )

    return retriever_tool, grouped_by_file


# ✅ Agent 대화 실행
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result["output"]


# ✅ 세션별 히스토리 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]


# ✅ 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])


# ✅ PDF 요약 생성
def summarize_pdf_grouped(grouped_docs: dict, llm: ChatOpenAI, max_chunks_per_file: int = 8) -> dict:
    """
    grouped_docs: { file_name: [Document, ...] }
    """
    summaries = {}
    for file_name, docs in grouped_docs.items():
        # 과도한 토큰 방지를 위해 앞부분 일부만 사용
        take = min(len(docs), max_chunks_per_file)
        contents = "\n\n".join(d.page_content for d in docs[:take])

        prompt = (
            "다음은 업로드된 PDF의 일부 내용입니다. 한국어로 간단하고 명료하게 핵심 요약을 작성하세요.\n"
            "- 5~8줄 이내 요약\n"
            "- 중요한 수치/키워드/정의는 **굵게** 표시\n"
            "- 문서의 전반적 목적과 주요 결론을 포함\n"
            "- 가능하면 (추정) 근거가 보이는 문장도 한 줄 포함\n\n"
            f"[문서: {file_name}] 내용:\n{contents}\n\n"
            "이제 [요약]만 출력하세요."
        )

        # ChatOpenAI.invoke는 문자열 프롬프트도 허용 (AIMessage 반환)
        ai_msg = llm.invoke(prompt)
        summaries[file_name] = ai_msg.content
    return summaries


# ✅ 메인 실행
def main():
    st.set_page_config(page_title="AI 비서 백수석-엔지니어 (RAG)", layout="wide", page_icon="🤖")

    with st.container():
        st.image('./chatbot_logo.png', use_container_width=True)
        st.markdown('---')
        st.title("안녕하세요! RAG를 활용한 'AI 비서 백수석-엔지니어' 입니다")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}
    if "pdf_grouped" not in st.session_state:
        st.session_state["pdf_grouped"] = {}
    if "pdf_summaries" not in st.session_state:
        st.session_state["pdf_summaries"] = {}

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키", placeholder="Enter Your API Key", type="password")
        st.markdown('---')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

        # ✅ PDF 요약 생성 버튼
        if pdf_docs and st.session_state.get("pdf_grouped"):
            if st.button("📌 PDF 요약 생성"):
                # LLM 인스턴스가 필요하므로 아래에서 생성된 llm을 다시 생성
                llm_for_summary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                with st.spinner("PDF 요약 생성 중..."):
                    st.session_state["pdf_summaries"] = summarize_pdf_grouped(
                        st.session_state["pdf_grouped"], llm_for_summary, max_chunks_per_file=8
                    )
                st.success("PDF 요약 생성 완료!")

    # ✅ 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
        os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        if pdf_docs:
            pdf_search_tool, grouped_by_file = load_pdf_files(pdf_docs)
            tools.append(pdf_search_tool)
            st.session_state["pdf_grouped"] = grouped_by_file  # 요약에 활용
        tools.append(search_web())

        # ✅ LLM 설정 (호환되는 파라미터 사용)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # ✅ 응답 형식 강화 (요약 표 + 출처 표)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "반드시 한국어로 답변하세요. 당신은 친절한 업무용 어시스턴트입니다. "
                    "당신의 이름은 `AI 비서 백수석-엔지니어`입니다. 대화 시작 시 짧게 자기소개하세요. "
                    "사용자의 요청이 PDF 기반이면 **반드시** `pdf_search` 툴을 먼저 호출해 관련 스니펫과 (파일명, 페이지)를 확보하세요. "
                    "PDF에서 정보를 찾을 수 없거나 질문에 '최신', '현재', '오늘'이 포함되면 `web_search` 툴을 사용하세요. "
                    "응답은 다음 형식을 따르세요:\n"
                    "1) 핵심 요약을 **간단한 표**로 제시\n"
                    "2) 필요한 경우 간단한 bullet 설명\n"
                    "3) 마지막에 **출처 표**를 제공합니다. PDF는 (파일명, 페이지), 웹은 (제목/도메인, 링크)\n"
                    "답변에는 항상 이모지를 포함하고, 친근하고 간결한 톤을 유지하세요."
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}\n\n Be sure to include emoji in your responses."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 입력창
        user_input = st.chat_input("어서오세요. 오늘은 어떤 도움을 드릴까요?")
        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            if session_history.messages:
                prev_msgs = [{"role": msg["role"], "content": msg["content"]} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            session_history.add_message({"role": "user", "content": user_input})
            session_history.add_message({"role": "assistant", "content": response})

        print_messages()

        # ✅ 생성된 PDF 요약 섹션 (있을 때만 표시)
        if st.session_state["pdf_summaries"]:
            st.markdown("---")
            st.subheader("📚 업로드된 PDF 요약")
            for fname, summ in st.session_state["pdf_summaries"].items():
                with st.expander(f"📝 {fname} 요약 보기"):
                    st.write(summ)

    else:
        st.warning("OpenAI API 키와 SerpAPI API 키를 입력하세요.")



