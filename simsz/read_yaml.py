import yaml
from typing import Union
import os

class YAMLOperator(object):

    def __init__(self, file_path= Union[None, str]):

        self.file_path = file_path 

    def parse_yaml(self):
        """
        Parse a YAML file and return a dictionary.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Dictionary containing the parsed YAML file.
        
        Raises:
            yaml.YAMLError: If the YAML file is not valid.
        """
        with open(self.file_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    
