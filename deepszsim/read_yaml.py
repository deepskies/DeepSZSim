"""
parsing a yaml file and returning a directory
"""

import yaml
from typing import Union
import os

class YAMLOperator:

    def __init__(self, file_path=os.path.join(os.path.dirname(__file__), "Settings", "config_simACTDR5.yaml")):

        self.file_path = file_path

    def parse_yaml(self):
        """
        Parse a YAML file and return a dictionary

        Parameters:
        -----------
        file_path: str
            Path to the YAML file.

        Returns:
        --------
        yaml.safe_load(): dict
            Dictionary containing the parsed YAML file.
            
        """
        with open(self.file_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
