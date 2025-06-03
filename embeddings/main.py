import click

from embed_doc import (
    generate_chroma_client,
    get_text_from_parsed_files,
    embed,
)


@click.command()
@click.option("--txt-location", help="txts file path", required=True)
def hello(txt_location: str) -> None:
    c_client = generate_chroma_client(
        company_name="Dixon Technologies (India) Ltd", year="2024"
    )
    for txt, pages in get_text_from_parsed_files(txt_location):
        embed(doc=list(txt), pages=list(pages), chroma_collection=c_client)


if __name__ == "__main__":
    hello()
