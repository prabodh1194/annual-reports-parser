from embeddings.from_parquet import from_parquet, store_embedding

pdf_path = "/Users/pbd/ar/2024/dixon.pdf"


# generate_embeddings( pdf_path=pdf_path, report_year="2024", company_name="Dixon Technologies" )

db = store_embedding(*from_parquet("~/ar/2024/dixon_embeddings.parquet"))
