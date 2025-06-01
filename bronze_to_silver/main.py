import click

from extract import extract_text_from_all_pages


@click.command()
@click.option("--pdf", help="pdf file path")
@click.option("--bronze", default="bronze-pdfs", help="pdf file path")
@click.option("--silver", default="silver-md", help="pdf file path")
def hello(pdf: str | None, bronze: str | None, silver: str | None) -> None:
    assert pdf
    extract_text_from_all_pages(pdf, bronze, silver)


if __name__ == "__main__":
    hello()
