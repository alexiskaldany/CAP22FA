from loguru import logger
import sys
logger.remove()
logger.add(
    sys.stdout,
    format="<light-yellow>{time:YYYY-MM-DD HH:mm:ss}</light-yellow> | <light-blue>{level}</light-blue> | <cyan>{message}</cyan> | <light-red>{function}: {line}</light-red>",
    level="DEBUG",
    backtrace=True,
    colorize=True,
)

from src.utils.configs import DATA_DIRECTORY
from src.utils.prepare_and_download import *
import click

@click.command()
@click.option('-d','--download', is_flag=True, help='Download the data')

def main(download)->None:
    if download:
        download_data(DATA_DIRECTORY)
        return
    
if __name__ == "__main__":
    main(prog_name="capstone")