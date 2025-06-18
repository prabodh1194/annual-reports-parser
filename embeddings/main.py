import click

from embeddings.embed_doc import (
    generate_chroma_client,
    embed,
    file_by_file_reader,
)


@click.command()
@click.option("--txt-location", help="txts file path", required=True)
def hello(txt_location: str) -> None:
    c_client = generate_chroma_client(
        company_name="Dixon Technologies (India) Ltd", year="2024"
    )
    for txt, idx in file_by_file_reader(folder_path=txt_location):
        if not txt:
            continue
        embed(doc=txt, pages=str(idx), chroma_collection=c_client)


if __name__ == "__main__":
    hello()
