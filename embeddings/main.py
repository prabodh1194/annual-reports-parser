import click

from embeddings.embed_doc import (
    generate_chroma_client,
    embed,
    chunk_token_generator_streaming,
)


@click.command()
@click.option("--txt-location", help="txts file path", required=True)
def hello(txt_location: str) -> None:
    c_client = generate_chroma_client(
        company_name="Dixon Technologies (India) Ltd", year="2024"
    )
    for idx, txt in enumerate(
        chunk_token_generator_streaming(folder_path=txt_location)
    ):
        embed(doc=txt, pages=str(idx), chroma_collection=c_client)


if __name__ == "__main__":
    hello()
