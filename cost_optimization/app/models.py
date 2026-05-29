from langchain_chroma import Chroma

from langchain_core.documents import Document

from langchain_core.prompts import (
ChatPromptTemplate,
)

from langchain_core.runnables import (
RunnablePassthrough,
)

from langchain_core.output_parsers import (
StrOutputParser,
)

from langchain_text_splitters import (
RecursiveCharacterTextSplitter,
)

from app.llm_factory import (
create_llm,
create_embeddings,
)

class LocalRAG:

def __init__(self):

    self.embeddings = create_embeddings()

    self.llm = create_llm("llama3")

    self.vectorstore = None

def build_knowledge_base(
    self,
    text: str,
    source: str = "document",
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
    )

    docs = [
        Document(
            page_content=text,
            metadata={"source": source},
        )
    ]

    chunks = splitter.split_documents(
        docs
    )

    self.vectorstore = (
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
    )

def create_chain(self):

    retriever = (
        self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
    )

    prompt = (
        ChatPromptTemplate.from_template(
            """

Answer using ONLY the context.

Context:
{context}

Question:
{question}

Answer:
"""
)
)

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    chain = (
        {
            "context": (
                retriever
                | format_docs
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | self.llm
        | StrOutputParser()
    )

    return chain
