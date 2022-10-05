import re
import sys
import pip
import json
import logging
import subprocess

logger = logging.getLogger(__name__)


class VersionService:

    def get_python_version(self):
        version = re.search(r'[\d.]+', sys.version)
        return version.group()

    def get_pip_version(self):
        return pip.__version__

    def get_library_version(self, library_name):
        list_files = subprocess.run(['pip3', 'show', library_name], capture_output=True)
        version = re.search(r'[\d.]+', list_files.stdout.decode())
        if version is None:
            return '-'
        else:
            return version.group()

    def get_pip_list(self, format: str = 'json'):
        """
        ref: https://minus9d.hatenablog.com/entry/2021/06/08/220614
        :return:
        """
        if format not in ['json', 'freeze', 'columns']:
            raise ValueError

        list_files = subprocess.run(['pip3', 'list', '--format', format], capture_output=True)

        match format:
            case 'json':
                return json.loads(list_files.stdout)
            case 'freeze':
                return list_files.stdout
            case _:
                pass
