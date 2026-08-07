import os
from tempfile import NamedTemporaryFile

import docx2txt


def process_file(uploaded_file):

    try:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")

        if uploaded_file.name.endswith(".docx"):
            with NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            text = docx2txt.process(tmp_path)
            os.remove(tmp_path)

            return text

        raise ValueError(
            "Unsupported file type. Please upload a .txt or .docx file."
        )

    except Exception as e:
        raise RuntimeError(f"File processing failed: {e}") from e