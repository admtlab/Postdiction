import sys
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

DATASET_INFO_FILE = "configuration_files/datasets_metadata.yaml"


"""
The code below here parses the data in the DATASET_INFO_FILE, which contains information about 
the attributes of a dataset. The `column_variable_get()` function should be used by 
outside classes to access this information.
"""


# Load in column information from csv file using the file and DATASET_INFO_FILE
def get_column_info_table(csv_file: str) -> dict:
    with open(DATASET_INFO_FILE, 'r') as file:
        yaml_data = yaml.safe_load(file)

    column_info_dict = {}
    file_name = csv_file.rsplit('/', 1)[-1]

    for column in yaml_data['datasets'][file_name]['columns']:
        column_name = column.get('name', '')
        column_type = column.get('type', '')
        column_size = column.get('size', '')
        column_info_dict[column_name] = [column_type, column_size]

    return column_info_dict

