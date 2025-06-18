import click

from bronze_to_silver.extract import extract_text_from_all_pages


@click.command()
@click.option("--pdf", help="pdf file path", required=True)
@click.option("--bronze", default="bronze-pdfs", help="pdf file path")
@click.option("--silver", default="silver-md", help="pdf file path")
def hello(pdf: str, bronze: str, silver: str) -> None:
    extract_text_from_all_pages(pdf, bronze, silver)


if __name__ == "__main__":
    hello()
