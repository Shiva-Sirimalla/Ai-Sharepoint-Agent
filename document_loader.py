import os
from pypdf import PdfReader
from docx import Document


class DocumentLoader:

    def load_documents(self, folder="documents"):

        documents = []

        if not os.path.exists(folder):
            return documents

        for file in os.listdir(folder):

            filepath = os.path.join(folder, file)

            try:

                if file.endswith(".pdf"):

                    pdf = PdfReader(filepath)

                    for page_num, page in enumerate(pdf.pages, start=1):

                        text = page.extract_text()

                        if text:

                            documents.append(
                                {
                                    "source": file,
                                    "page": page_num,
                                    "content": text
                                }
                            )

                elif file.endswith(".docx"):

                    doc = Document(filepath)

                    text = "\n".join(
                        [p.text for p in doc.paragraphs]
                    )

                    documents.append(
                        {
                            "source": file,
                            "page": 1,
                            "content": text
                        }
                    )

                elif file.endswith(".txt"):

                    with open(
                        filepath,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        text = f.read()

                    documents.append(
                        {
                            "source": file,
                            "page": 1,
                            "content": text
                        }
                    )

            except Exception as e:

                print(f"Error loading {file}: {e}")

        return documents

    def get_stats(self, folder="documents"):

        total_files = 0
        total_pages = 0

        if not os.path.exists(folder):

            return {
                "files": 0,
                "pages": 0
            }

        for file in os.listdir(folder):

            filepath = os.path.join(folder, file)

            try:

                if file.endswith(".pdf"):

                    pdf = PdfReader(filepath)

                    total_files += 1
                    total_pages += len(pdf.pages)

                elif file.endswith(".docx"):

                    total_files += 1
                    total_pages += 1

                elif file.endswith(".txt"):

                    total_files += 1
                    total_pages += 1

            except Exception as e:

                print(f"Error reading stats from {file}: {e}")

        return {
            "files": total_files,
            "pages": total_pages
        }
    
print("DOCUMENT LOADER UPDATED VERSION LOADED")    