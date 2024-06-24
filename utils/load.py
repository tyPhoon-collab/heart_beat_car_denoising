from logging import warning
from dotenv import load_dotenv


def load_local_dotenv():
    ret = load_dotenv(override=True)
    if not ret:
        warning("Could not load .env file.")
