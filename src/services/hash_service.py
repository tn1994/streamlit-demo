import hashlib


def make_hashes(password: str) -> str:
    """
    ref: https://zenn.dev/lapisuru/articles/3ae6dd82e36c29a27190
    :param password:
    :return:
    """
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password: str, hashed_text: str) -> str or bool:
    """
    ref: https://zenn.dev/lapisuru/articles/3ae6dd82e36c29a27190
    :param password:
    :param hashed_text:
    :return:
    """
    if make_hashes(password) == hashed_text:
        return True
    else:
        return False
