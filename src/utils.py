import os
import sys

import numpy as np
import pandas as pd
import dill

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException

def save_object(obj, file_path):
    """
    This function is used to save the object to the file path.
    :param obj: object to be saved
    :param file_path: file path where the object will be saved
    :return: None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException("Error occurred in save_object", e)
