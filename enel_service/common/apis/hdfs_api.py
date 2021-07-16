import os
import jsonpickle
from hdfs import InsecureClient
import pandas as pd
import logging
import json
import re
import yaml
import io
import torch
import dill

from .api_interface import StorageApi
from ..configuration import HdfsSettings


class HdfsApi(StorageApi):
    def __init__(self):

        self.settings: HdfsSettings = HdfsSettings.get_instance()

        self.output_dir = self.remove_leading_slash(self.settings.hdfs_output_dir)
        self.client = InsecureClient(self.settings.hdfs_endpoint)

        self.__hash_target__ = " | ".join([f"key={k},value={v}" for k, v in self.settings.dict().items()])

    def __hash__(self):
        return len(self.__hash_target__)

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    @staticmethod
    def remove_leading_slash(s: str):
        return re.sub(r"^/+", "", s)

    def _build_path(self, target_dir=None):
        if target_dir is None:
            return os.path.join("/", self.output_dir)
        else:
            target_dir = self.remove_leading_slash(target_dir)
            return os.path.join("/", self.output_dir, target_dir)
        
    def load(self, my_file_name, target_dir=None, is_absolute_path=False,  **kwargs):
        """
        
        Load a file from HDFS.
        
        :param str my_file_name: The name of the desired file (including file-ending).
        :param str target_dir: The sub-directory of the desired file, default is directory defined globally via configuration-file.
        :param boolean is_absolute_path: Flag to label whether my_file_name is absolute path
        :param \*\*kwargs: Any other arguments, will be passed to pandas.read_csv if target file is csv-file.
        
        """
        if not is_absolute_path:
            target_dir = self._build_path(target_dir=target_dir)
        my_file_name = self.remove_leading_slash(my_file_name)
        
        data = None
        error = None

        encoding = "utf-8" if ".pt" not in my_file_name else None

        try:
            with self.client.read(os.path.join(target_dir, my_file_name), encoding=encoding) as reader:
                if ".csv" in my_file_name:
                    data = pd.read_csv(reader, **kwargs)
                elif ".json" in my_file_name:
                    data = json.load(reader)
                elif ".yaml" in my_file_name:
                    data = yaml.safe_load(reader)
                elif ".pt" in my_file_name:
                    buff = io.BytesIO(reader.read())
                    if "torchscript" in my_file_name:
                        data = torch.jit.load(buff, map_location=torch.device("cpu"))
                    else:
                        data = torch.load(buff, map_location=torch.device("cpu"))
                elif ".pkl" in my_file_name:
                    data = dill.load(reader)
                else:
                    data = reader.read()
        except BaseException as e:
            logging.error(e)
            error = str(e)
        return data, error        

    def list_dir(self, target_dir=None, is_absolute_path=False, **kwargs):
        """
        
        Get all names of files located in an HDFS directory.
        
        :param str target_dir: The sub-directory of the desired file, default is directory defined globally via configuration-file.
        :param is_absolute_path: flag to determine whether hdfs_dir is absolute path

        """
        if is_absolute_path is False:
            target_dir = self._build_path(target_dir=target_dir)
        
        file_names: list = []
        error = None
        try:
            file_names = self.client.list(target_dir, **kwargs)
        except BaseException as e:
            logging.error(e)
            error = str(e)               
        return file_names, error  

    def download(self, file_name, hdfs_dir, local_dir, is_absolute_path=False, override=False, **kwargs):
        """
        download file from hdfs
        :param file_name: file name
        :param hdfs_dir: hdfs directory
        :param local_dir: local directory
        :param is_absolute_path: flag to determine whether hdfs_dir is absolute path
        :param override: flag to override the local file if it exists
        :param kwargs:
        :return:
        """
        if not is_absolute_path:
            hdfs_path = self._build_path(target_dir=hdfs_dir)
        else:
            hdfs_path = hdfs_dir
        hdfs_path = os.path.join(hdfs_path, file_name)

        local_path = os.path.join(local_dir, file_name)

        override and os.path.isfile(local_path) and os.remove(local_path)
        
        response_path = None
        try:
            response_path = self.client.download(hdfs_path, local_path, **kwargs)       
        except BaseException as e:
            logging.error(e)
        return response_path

    def delete(self, my_file_name, target_dir=None, **kwargs):

        target_dir = self._build_path(target_dir=target_dir)
        my_file_name = self.remove_leading_slash(my_file_name)

        success = False
        try:
            success = self.client.delete(os.path.join(target_dir, my_file_name), **kwargs)
        except BaseException as e:
            logging.error(e)
        return success

    def save(self, my_file, my_file_name, mode="w", target_dir=None, multi_layer_folder=False,  **kwargs):
        """
        
        Write a file to HDFS.
        
        :param dict|DataFrame my_file: The actual data that needs to be written.
        :param str my_file_name: The name of the file (including file-ending).
        :param str mode: The mode of the operation ('w'= write or 'a'=append).
        :param str target_dir: The sub-directory of the file, default is directory defined globally via configuration-file.
        :param boolean multi_layer_folder: enables 3+ layer folder hierarchy structure
        :param \*\*kwargs: Any other arguments, will be passed to pandas.to_csv if file is csv-file.
        
        """
        
        target_dir = self._build_path(target_dir=target_dir)
        if multi_layer_folder is False:
            # enable multi folder hierarchy
            my_file_name = re.sub("\/", "", my_file_name)

        error = None

        encoding = "utf-8" if ".pt" not in my_file_name else None

        try:
            with self.client.write(os.path.join(target_dir, my_file_name),
                                   encoding=encoding,
                                   append=mode == "a",
                                   overwrite=mode == "w") as writer:
                if ".csv" in my_file_name:
                    try:
                        my_file.to_csv(writer, mode=mode, **kwargs)
                    except AttributeError:
                        writer.write(my_file)
                elif ".json" in my_file_name:
                    encoded_content = jsonpickle.encode(my_file)
                    writer.write(encoded_content)
                elif ".yaml" in my_file_name:
                    yaml.safe_dump_all(my_file, writer)
                elif ".pt" in my_file_name and "torchscript" in my_file_name:
                    torch.jit.save(my_file, writer)
                elif ".pt" in my_file_name:
                    torch.save(my_file, writer)
                elif ".pkl" in my_file_name:
                    dill.dump(my_file, writer)
                else:
                    writer.write(my_file)
        except BaseException as e:
            logging.error(e)
            error = str(e)
        return error

    def exists_file(self, filename, directory):
        """
        check file exists or not
        :param filename:
        :param directory:
        :return: True: file exists; False: otherwise
        """
        target_dir = self._build_path(target_dir=directory)
        my_file_name = self.remove_leading_slash(filename)
        status = self.client.status(os.path.join(target_dir, my_file_name), strict=False)
        return status is not None

    def content(self, path):
        """
        check information of given path
        :param path: hdfs path
        :return:
        """
        return self.client.content(path, strict=False)
