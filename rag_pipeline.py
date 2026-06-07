from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from vector_store import embedding_model


class RAGPipeline:

    def __init__(self):

        self.chunk_count = 0

    def create_vector_db(self, docs):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = []
        metadata = []

        for doc in docs:

            split_chunks = splitter.split_text(
                doc["content"]
            )

            for chunk in split_chunks:

                chunks.append(chunk)

                metadata.append(
                    {
                        "source": doc["source"],
                        "page": doc["page"]
                    }
                )

        self.chunk_count = len(chunks)

        db = FAISS.from_texts(
            chunks,
            embedding_model,
            metadatas=metadata
        )

        db.save_local("vector_db")

    def load_db(self):

        return FAISS.load_local(
            "vector_db",
            embedding_model,
            allow_dangerous_deserialization=True
        )