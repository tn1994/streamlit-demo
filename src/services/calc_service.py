import logging

logger = logging.getLogger(__name__)


class CalcService:

    def __init__(self):
        pass

    def get_eval(self, text: str, variables: str = None):
        try:
            if not isinstance(text, str):
                raise TypeError
            if variables is not None:
                if not isinstance(variables, str):
                    raise TypeError
                variables_dict = {}
                exec(variables, {}, variables_dict)
                result = eval(text, {}, variables_dict)
            else:
                result = eval(text)
        except Exception as e:
            logger.error(e)
            raise e
        else:
            return result
