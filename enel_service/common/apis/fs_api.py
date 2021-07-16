import logging
import yaml

from enel_service.common.apis.api_interface import StorageApi


class FsApi(StorageApi):

    def load(self, path_to_file: str):
        data = None
        error = None

        try:
            with open(path_to_file, "r") as reader:
                if ".yaml" in path_to_file:
                    data = list(yaml.safe_load_all(reader))
                else:
                    data = reader.read()
        except BaseException as e:
            logging.error(e)
            error = str(e)
        return data, error

    def save(self, my_file, path_to_file: str):

        error = None

        try:
            with open(path_to_file, "w") as writer:
                if ".yaml" in path_to_file:
                    yaml.safe_dump_all(my_file, writer)
                else:
                    writer.write(my_file)
        except BaseException as e:
            logging.error(e)
            error = str(e)
        return error
