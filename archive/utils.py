import time
from logger import logging


def get_with_retry(func, *args, **kwargs):
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            logger.error(f"Error: {e}. Retrying in 10 seconds... (attempt {retries})")
            time.sleep(10)
    raise RuntimeError("Failed to invoke the function after multiple retries.")
