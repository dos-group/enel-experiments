import logging
import re
import subprocess
from datetime import datetime
from os import path

import yaml
from kubernetes import client
from kubernetes.client import V1PodList

from .api_interface import RESTApi
from ..configuration import KubernetesSettings
from ..kubernetes_client_utils import KubernetesApiHelper, KubernetesApiResponse


class KubernetesApi(RESTApi):
    def __init__(self):

        self.settings: KubernetesSettings = KubernetesSettings.get_instance()

        self.endpoint: str = self.settings.kubernetes_endpoint
        self.namespace: str = self.settings.kubernetes_namespace

        configuration = client.Configuration()
        configuration.api_key['authorization'] = self.settings.kubernetes_api_key
        configuration.api_key_prefix['authorization'] = self.settings.kubernetes_api_key_prefix
        configuration.host = self.endpoint
        configuration.verify_ssl = False

        self.client = client.ApiClient(configuration)
        self.core_v1_api = client.CoreV1Api(self.client)

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def post(self, job_def_file, namespace=None):
        namespace = namespace or self.namespace
        response = None
        error = None
        try:
            response = create_from_yaml(self.client, job_def_file, namespace=namespace)
        except BaseException as e:
            logging.error(e)
            error = str(e)
        return response, error

    def get_pod(self, namespace="default", **kwargs) -> KubernetesApiResponse[V1PodList]:
        """
        get pod information
        :param namespace:
        :param kwargs:
        :return: KubernetesApiResponse[V1PodList]
        """
        api_response = KubernetesApiHelper.call_k8s_api(self.core_v1_api.list_namespaced_pod, namespace=namespace, **kwargs)
        return api_response

    def delete_pod(self, name, namespace="default"):
        """
        delete pod with the given pod name and namespace
        :param name:
        :param namespace:
        :return: bool
        """
        api_response = KubernetesApiHelper.call_k8s_api(self.core_v1_api.delete_namespaced_pod,
                                                        name=name,
                                                        namespace=namespace)
        return api_response.success


def update_dict_func(source_dict: dict, update_dict: dict, check_existence=True):
    source_dict = source_dict or {}
    update_dict = update_dict or {}

    for key, value in update_dict.items():

        if check_existence and key not in source_dict:
            continue

        if isinstance(value, dict):
            source_dict[key] = update_dict_func(source_dict.get(key, {}), value, check_existence=check_existence)
        elif value is not None:
            source_dict[key] = value

    return source_dict


def generate_template_code(source_folder, source_values_file, check: bool = False,
                           release_name=f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
    template_code = None
    error = None
    try:
        cmd = ["helm", "template", release_name, source_folder, "--values", source_values_file]
        process = subprocess.run(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 check=check)
        if process.stderr:
            stderr = process.stderr
            logging.error(stderr.decode("utf-8") if not isinstance(stderr, str) else stderr)
        else:
            stdout = process.stdout
            yaml_string = stdout.decode("utf-8") if not isinstance(stdout, str) else stdout
            template_code = list(yaml.safe_load_all(yaml_string))

    except BaseException as e:
        logging.error(e)
        error = str(e)
    return template_code, error


############################### BELOW: copied and adapted from ##############################################
### https://github.com/kubernetes-client/python/blob/master/kubernetes/utils/create_from_yaml.py [bbdfb73] ##
#############################################################################################################
def create_from_yaml(
        k8s_client,
        yaml_file,
        verbose=False,
        namespace="default",
        **kwargs):
    with open(path.abspath(yaml_file)) as f:
        yml_document_all = yaml.safe_load_all(f)

        failures = []
        k8s_objects = []
        for yml_document in yml_document_all:
            try:
                created = create_from_dict(k8s_client, yml_document, verbose,
                                           namespace=namespace,
                                           **kwargs)
                k8s_objects.append(created)
            except FailToCreateError as failure:
                failures.extend(failure.api_exceptions)
        if failures:
            raise FailToCreateError(failures)

        return k8s_objects


def create_from_dict(k8s_client, data, verbose=False, namespace='default',
                     **kwargs):
    # If it is a list type, will need to iterate its items
    api_exceptions = []
    k8s_objects = []

    if "List" in data["kind"]:
        # Could be "List" or "Pod/Service/...List"
        # This is a list type. iterate within its items
        kind = data["kind"].replace("List", "")
        for yml_object in data["items"]:
            # Mitigate cases when server returns a xxxList object
            # See kubernetes-client/python#586
            if kind != "":
                yml_object["apiVersion"] = data["apiVersion"]
                yml_object["kind"] = kind
            try:
                created = create_from_yaml_single_item(
                    k8s_client, yml_object, verbose, namespace=namespace,
                    **kwargs)
                k8s_objects.append(created)
            except client.rest.ApiException as api_exception:
                api_exceptions.append(api_exception)
    else:
        # This is a single object. Call the single item method
        try:
            created = create_from_yaml_single_item(
                k8s_client, data, verbose, namespace=namespace, **kwargs)
            k8s_objects.append(created)
        except client.rest.ApiException as api_exception:
            api_exceptions.append(api_exception)

    # In case we have exceptions waiting for us, raise them
    if api_exceptions:
        raise FailToCreateError(api_exceptions)

    return k8s_objects


def create_from_yaml_single_item(
        k8s_client, yml_object, verbose=False, **kwargs):
    group, _, version = yml_object["apiVersion"].partition("/")
    if version == "":
        version = group
        group = "core"
    k8s_api = getattr(client, "CustomObjectsApi")(k8s_client)
    # Replace CamelCased action_type into snake_case
    kind = yml_object["kind"]
    kind = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', kind)
    kind = re.sub('([a-z0-9])([A-Z])', r'\1_\2', kind).lower()

    kwargs["group"] = group
    kwargs["version"] = version
    kwargs["plural"] = re.sub("[^a-zA-Z]+", "", kind) + "s"

    resp = getattr(k8s_api, "create_namespaced_custom_object")(
        body=yml_object, **kwargs)

    if verbose:
        msg = "{0} created.".format(kind)
        if hasattr(resp, 'status'):
            msg += " status='{0}'".format(str(resp.status))
        print(msg)
    return resp


class FailToCreateError(Exception):
    """
    An exception class for handling error if an error occurred when
    handling a yaml file.
    """

    def __init__(self, api_exceptions):
        self.api_exceptions = api_exceptions

    def __str__(self):
        msg = ""
        for api_exception in self.api_exceptions:
            msg += "Error from server ({0}): {1}".format(
                api_exception.reason, api_exception.body)
        return msg
