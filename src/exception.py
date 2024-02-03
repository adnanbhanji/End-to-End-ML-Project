import sys
import traceback
from logger import logging 

def error_message_detail(error, tb):
    file_name = tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, tb):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, tb)
    
    def __str__(self):
        return self.error_message

if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as e:
        tb = e.__traceback__
        logging.info('Error occurred')
        raise CustomException(e, tb) from None
