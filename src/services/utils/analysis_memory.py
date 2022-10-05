import psutil


def get_memory_state_percent() -> float:
    """

    :return: percent
    """
    memory_state = psutil.virtual_memory()
    return memory_state.percent
