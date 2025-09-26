import os
import tempfile
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage


# --------- 전역 설정 ----------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ✅ SerpAPI 검색 툴
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
        description="실시간 뉴스 및 웹 정보를 검색합니다. (제목/출처/링크/스니펫 반환)"
    )


# ✅ PDF 업로드 → 벡터DB → 검색 툴 생성 (+ 메타데이터 보강)
def load_pdf_files(uploaded_files):
    all_documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        for d in documents:
            d.metadata["file_name"] = uploaded_file.name  # 원본 업로드 파일명

        all_documents.extend(documents)

    # 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    # 파일별 그룹핑 (요약용)
    grouped_by_file = {}
    for d in split_docs:
        fname = d.metadata.get("file_name", "unknown.pdf")
        grouped_by_file.setdefault(fname, []).append(d)

    # 벡터DB 생성
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    # PDF 검색 툴: 스니펫 + (파일명, 페이지)
    def run_pdf_search(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "PDF에서 관련 정보를 찾지 못했습니다."
        lines = []
        for i, d in enumerate(docs[:5]):
            file_name = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", ""))
            page_meta = d.metadata.get("page")
            page_disp = page_meta + 1 if isinstance(page_meta, int) else page_meta
            snippet = d.page_content.strip().replace("\n", " ")
            lines.append(f"{i+1}. {snippet}\n   (출처: {file_name}, p.{page_disp})")
        return "\n".join(lines)

    retriever_tool = Tool(
        name="pdf_search",
        func=run_pdf_search,
        description="업로드된 PDF에서 검색합니다. (스니펫 + 출처: 파일명, 페이지)"
    )

    return retriever_tool, grouped_by_file


# ✅ PDF 요약
def summarize_pdf_grouped(grouped_docs: dict, llm: ChatOpenAI, max_chunks_per_file: int = 8) -> dict:
    summaries = {}
    for file_name, docs in grouped_docs.items():
        take = min(len(docs), max_chunks_per_file)
        contents = "\n\n".join(d.page_content for d in docs[:take])

        prompt = (
            "다음은 업로드된 PDF의 일부 내용입니다. 한국어로 간단하게 핵심 요약을 작성하세요.\n"
            "- 5~8줄 요약\n- 중요한 수치/키워드/정의는 **굵게**\n- 문서 목적과 주요 결론 포함\n"
            f"[문서: {file_name}] 내용:\n{contents}\n\n"
            "이제 [요약]만 출력하세요."
        )
        ai_msg = llm.invoke(prompt)
        summaries[file_name] = ai_msg.content
    return summaries


# ✅ 세션 히스토리
def get_session_history(session_id: str) -> ChatMessageHistory:
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}
    if session_id not in st.session_state["session_history"]:
        st.session_state["session_history"][session_id] = ChatMessageHistory()
    return st.session_state["session_history"][session_id]


# ✅ 이전 메시지 출력 (UI)
def print_messages():
    if "messages" not in st.session_state:
        return
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])


# ✅ 메인
def main():
    # 반드시 첫 Streamlit 호출
    st.set_page_config(page_title="AI 비서 백수석-엔지니어 (RAG)", layout="wide", page_icon="🤖")

    # 상태 초기화
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pdf_grouped", {})
    st.session_state.setdefault("pdf_summaries", {})

    # 헤더
    with st.container():
        try:
            st.image("./chatbot_logo.png", width="stretch", use_container_width=True,use_column_width="always")
        except Exception:
            st.info("로고 이미지를 찾지 못했습니다. (chatbot_logo.png)")
        st.markdown('---')
        st.title("안녕하세요! RAG를 활용한 'AI 비서 백수석-엔지니어' 입니다 👋")

    # 사이드바
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키", placeholder="Enter Your API Key", type="password")
        st.markdown('---')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

        # 요약 버튼은 파일 업로드 이후 노출
        if pdf_docs:
            if st.button("📌 PDF 요약 생성"):
                if not st.session_state.get("pdf_grouped"):
                    st.warning("먼저 키 입력 후 PDF를 로딩하세요.")
                else:
                    llm_for_summary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    with st.spinner("PDF 요약 생성 중..."):
                        st.session_state["pdf_summaries"] = summarize_pdf_grouped(
                            st.session_state["pdf_grouped"], llm_for_summary, max_chunks_per_file=8
                        )
                    st.success("PDF 요약 생성 완료!")

    # 본문 — 키 미입력 시에도 기본 UI 보이도록 먼저 렌더
    st.markdown("### 대화")
    user_input = st.chat_input("어서오세요. 오늘은 어떤 도움을 드릴까요?")

    # 키 확인 후 에이전트/툴 준비
    agent_executor = None
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]
        os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API"]

        tools = []
        if pdf_docs:
            pdf_search_tool, grouped_by_file = load_pdf_files(pdf_docs)
            tools.append(pdf_search_tool)
            st.session_state["pdf_grouped"] = grouped_by_file
        tools.append(search_web())

        # LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "반드시 한국어로 답변하세요. 당신은 친절한 업무용 어시스턴트입니다. "
                 "당신의 이름은 `AI 비서 백수석-엔지니어`입니다. 대화 시작 시 짧게 자기소개하세요. "
                 "PDF 기반이면 `pdf_search`를 우선 사용하고, '최신/현재/오늘' 질문이면 `web_search`를 사용하세요. "
                 "응답 형식: 1) 핵심 요약 표 2) 필요 시 짧은 bullet 3) 마지막에 출처 표. "
                 "항상 이모지를 포함하세요."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}\n\nBe sure to include emoji in your responses."),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

    # 대화 처리
    session_id = "default_session"
    session_history = get_session_history(session_id)

    if user_input:
        if agent_executor:
            with st.spinner("답변 생성 중..."):
                result = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": session_history.messages  # <-- 핵심 수정
                })
                response = result["output"]
        else:
            response = "⚠️ 먼저 사이드바에 OpenAI/SerpAPI 키를 입력해주세요."

        # UI 기록
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # LangChain 히스토리 기록 (BaseMessage)
        session_history.add_user_message(user_input)
        session_history.add_ai_message(response)

    # 메시지 출력
    print_messages()

    # PDF 요약 출력
    if st.session_state["pdf_summaries"]:
        st.markdown("---")
        st.subheader("📚 업로드된 PDF 요약")
        for fname, summ in st.session_state["pdf_summaries"].items():
            with st.expander(f"📝 {fname} 요약 보기"):
                st.write(summ)


if __name__ == "__main__":
    main()
