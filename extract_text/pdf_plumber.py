import pdfplumber


def extract_text_from_pdf(pdf_path: str):
    text_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            words = sorted(
                words, key=lambda w: (w["top"], w["x0"])
            )  # Sort by Y, then X

            line_texts: dict = {}
            for word in words:
                line_y = round(word["top"])  # Approximate line position
                if line_y not in line_texts:
                    line_texts[line_y] = []
                line_texts[line_y].append(word["text"])

            page_text = "\n".join(
                [" ".join(line_texts[y]) for y in sorted(line_texts.keys())]
            )
            text_data.append(page_text)

    return "\n\n".join(text_data)
