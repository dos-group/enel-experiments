import asyncio
import functools
import logging
import datetime
from datetime import date
from typing import List, Any, Optional

import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel

from .api_interface import DatabaseApi
from ..configuration import MongoSettings


def handle_connection(func):
    @functools.wraps(func)
    async def wrapper(self, collection_name: str, *args, **kwargs):
        # establish connection
        await self.connect()

        collection = self.database.get_collection(collection_name)
        result = None
        if collection is not None:
            new_args: tuple = tuple([collection_name]) + args
            result = await func(self, *new_args, **kwargs)
        else:
            logging.error(f"Collection '{collection_name}' does not exists in database '{self.mongodb_database}'.")

        # tear down connection
        await self.disconnect(kwargs.get("session", None) is not None)
        return result

    return wrapper


class MongoApi(DatabaseApi):
    def __init__(self):
        self.settings: MongoSettings = MongoSettings.get_instance()
        self.mongodb_database: str = self.settings.mongodb_database

        self.connection_string: str = ("mongodb://"
                                       f"{self.settings.mongodb_username}:{self.settings.mongodb_password}@"
                                       f"{self.settings.mongodb_endpoint}:{self.settings.mongodb_port}"
                                       f"/{self.settings.mongodb_database}?authSource=admin")

        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[Any] = None

        self.debug_log = lambda coll, msg, err=None: logging.debug((f"DB={self.mongodb_database}, "
                                                                    f"Collection={coll}: "
                                                                    f"{msg}"
                                                                    f"{'' if err is None else f'(Root Error: {err})'}"))

        self.error_log = lambda coll, msg, err=None: logging.error((f"DB={self.mongodb_database}, "
                                                                    f"Collection={coll}: "
                                                                    f"{msg} "
                                                                    f"{'' if err is None else f'(Root Error: {err})'}"))
        self.exception_log = lambda coll, msg, exc, err=None: logging.error((f"DB={self.mongodb_database}, "
                                                                             f"Collection={coll}: "
                                                                             f"{msg} "
                                                                             f"{'' if err is None else f'(Root Error: {err})'}"),
                                                                            exc_info=exc)

    async def connect(self):
        self.get_client()
        logging.debug(f"Connecting to database '{self.mongodb_database}'...")
        self.database = self.client[self.mongodb_database]

    async def disconnect(self, active_session: bool = False):
        if not active_session:
            await self.close_client()
        logging.debug(f"Disconnecting from database '{self.mongodb_database}'...")
        self.database = None

    def get_client(self) -> AsyncIOMotorClient:
        if self.client is None:
            self.client = AsyncIOMotorClient(self.connection_string)
        return self.client

    async def close_client(self):
        if self.client is not None:
            close_func = self.client.close()
            if asyncio.iscoroutine(close_func):
                await close_func
        self.client = None

    @staticmethod
    async def create_indexes():
        mongo_api: MongoApi = MongoApi()

        logging.info("Create DB indexes, if not already done...")

        async with await mongo_api.get_client().start_session() as s:
            await mongo_api.connect()

            job_collection = mongo_api.database.get_collection(
                mongo_api.settings.mongodb_job_execution_collection
            )

            await job_collection.create_indexes([
                IndexModel([("application_id", pymongo.ASCENDING),
                            ("job_id", pymongo.ASCENDING)], name="job_index_compound_sort", unique=True),
                IndexModel([("application_execution_id", pymongo.ASCENDING),
                            ("application_id", pymongo.ASCENDING),
                            ("job_id", pymongo.ASCENDING)], name="job_index_compound_search", unique=True),
                IndexModel([("application_execution_id", pymongo.ASCENDING),
                            ("application_signature", pymongo.ASCENDING),
                            ("job_id", pymongo.ASCENDING),
                            ("start_time", pymongo.ASCENDING),
                            ("end_time", pymongo.ASCENDING)], name="job_index_compound_pred", unique=True),
                IndexModel([("application_signature", pymongo.ASCENDING),
                            ("job_id", pymongo.ASCENDING),
                            ("start_time", pymongo.ASCENDING),
                            ("end_time", pymongo.ASCENDING)], name="job_index_compound_succ")
            ], session=s)

            await mongo_api.disconnect(active_session=True)

        await mongo_api.close_client()

    @handle_connection
    async def find(self, collection_name: str, *args, catch_error: bool = True, **kwargs):
        collection = self.database.get_collection(collection_name)
        results: list = []
        try:
            results = await collection.find(*args, **kwargs).to_list(length=None)
            self.debug_log(collection_name, f"Found {len(results)} documents in '{collection_name}' "
                                            f"with options '{args, kwargs}'")
        except BaseException as e:
            self.error_log(collection_name, f"Could not execute 'find'-command in '{collection_name}' "
                                            f"with options '{args, kwargs}'", err=e)
            if not catch_error:
                raise e
        return results

    @handle_connection
    async def find_one(self, collection_name: str, filter_dict: dict, *args, catch_error: bool = True, **kwargs):
        """
        find one document according to filter_dict
        :param collection_name:
        :param filter_dict: filter in dict type
        :param args:
        :param catch_error:
        :param kwargs:
        :return:
        """
        collection = self.database.get_collection(collection_name)
        result = None
        try:
            result = await collection.find_one(filter_dict, *args, **kwargs)
            self.debug_log(collection_name, f"Found a document {result} in '{collection_name}' "
                                            f"with options '{args, kwargs}'")
        except BaseException as e:
            self.error_log(collection_name, f"Could not execute 'find_one'-command in '{collection_name}' "
                                            f"with options '{args, kwargs}'",
                           err=e)
            if not catch_error:
                raise e
        return result

    @handle_connection
    async def insert(self, collection_name: str, documents: List[Any], catch_error: bool = True, **kwargs):
        collection = self.database.get_collection(collection_name)
        result_ids: list = []
        try:
            created_at: date = datetime.datetime.now()
            documents = [{**element, "created_at": created_at} for element in documents]
            result = await collection.insert_many(documents)
            if result.acknowledged:
                result_ids = [str(object_id) for object_id in result.inserted_ids]
                self.debug_log(collection_name, f"Inserted {len(result_ids)} documents in '{collection_name}' "
                                                f"with inserted_ids={result_ids}")
            else:
                raise Exception("No acknowledged operation")
        except BaseException as e:
            self.error_log(collection_name, f"Could not execute 'insert'-command in '{collection_name}' "
                                            f"with documents '{documents}'", err=e)
            if not catch_error:
                raise e
        return result_ids

    async def insert_one(self, collection_name: str, document: Any, **kwargs):
        result = await self.insert(collection_name, [document], **kwargs)
        if result is not None and len(result):
            return result[0]
        else:
            return None

    @handle_connection
    async def update(self, collection_name: str,
                     filter_dict: dict, update_dict: dict, *args, catch_error: bool = True, **kwargs):
        collection = self.database.get_collection(collection_name)
        acknowledged: bool = False
        try:
            updated_at: date = datetime.datetime.now()
            update_dict["$set"] = {**update_dict.get("$set", {}), "updated_at": updated_at}
            result = await collection.update_many(filter_dict, update_dict, *args, **kwargs)
            self.debug_log(collection_name, (f"Matched {result.matched_count} documents, "
                                             f"modified {result.modified_count} documents"))
            acknowledged = result.acknowledged
        except BaseException as e:
            self.error_log(collection_name, f"Could not execute 'update'-command in '{collection_name}' "
                                            f"with options '{args, kwargs}'", err=e)
            if not catch_error:
                raise e
        return acknowledged

    @handle_connection
    async def update_one(self, collection_name: str, filter_dict: dict,
                         update_dict: dict, *args, catch_error: bool = True, create: bool = False, **kwargs):
        """
        update one document
        :param collection_name: collection name
        :param filter_dict: filter in type dict
        :param update_dict: update statement in dict
        :param args:
        :param catch_error:
        :param create:
        :param kwargs:
        :return:
        """
        collection = self.database.get_collection(collection_name)
        acknowledged: bool = False
        try:
            updated_at: date = datetime.datetime.now()
            update_dict["$set"] = {**update_dict.get("$set", {}), "updated_at": updated_at}
            if create and kwargs.get("upsert", False):
                update_dict["$set"] = {**update_dict.get("$set", {}), "created_at": updated_at}
            result = await collection.update_one(filter_dict, update_dict, *args, **kwargs)
            self.debug_log(collection_name, (f"Matched {result.matched_count} documents, "
                                             f"modified {result.modified_count} documents"))
            acknowledged = result.acknowledged
        except BaseException as e:
            self.exception_log(collection_name, f"Could not execute 'update'-command in '{collection_name}' "
                                                f"with options '{args, kwargs}'",
                               exc=e)
            if not catch_error:
                raise e
        return acknowledged

    @handle_connection
    async def delete(self, collection_name: str, *args, catch_error: bool = True, **kwargs):
        collection = self.database.get_collection(collection_name)
        acknowledged: bool = False
        try:
            result = await collection.delete_many(*args, **kwargs)
            self.debug_log(collection_name, f"Deleted {result.deleted_count} documents")
            acknowledged = result.acknowledged
        except BaseException as e:
            self.error_log(collection_name, f"Could not execute 'delete'-command in '{collection_name}' "
                                            f"with options '{args, kwargs}'", err=e)
            if not catch_error:
                raise e
        return acknowledged

    @handle_connection
    async def aggregate(self, collection_name: str, *args, catch_error: bool = True, **kwargs):
        collection = self.database.get_collection(collection_name)
        results: list = []
        try:
            results = await collection.aggregate(*args, **kwargs).to_list(length=None)
            self.debug_log(collection_name, f"Aggregation result of length {len(results)} on '{collection_name}' "
                                            f"with options '{args, kwargs}'")
        except BaseException as e:
            self.error_log(collection_name, f"Could not execute 'aggregate'-command on '{collection_name}' "
                                            f"with options '{args, kwargs}'", err=e)
            if not catch_error:
                raise e
        return results
