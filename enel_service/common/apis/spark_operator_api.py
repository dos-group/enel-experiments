import kubernetes
from kubernetes.client import V1PodList

from enel_service.common.apis.kubernetes_api import KubernetesApi
from enel_service.common.kubernetes_client_utils import KubernetesApiHelper, KubernetesApiResponse


class SparkOperatorApi:
    """
    Spark Operator API
    """

    def __init__(self):
        # custom resource group
        self.cr_group = "sparkoperator.k8s.io"
        # custom resource version
        self.cr_version = "v1beta2"
        # custom resource plural
        self.cr_plural = "sparkapplications"
        self.custom_objects_api = kubernetes.client.CustomObjectsApi(KubernetesApi().client)

    def create_spark_application(self, application_description: dict):
        """
        create spark application from application description in dict type
        :param application_description: application description in dict
        :return:
        """
        return KubernetesApiHelper.call_k8s_api(self.custom_objects_api.create_namespaced_custom_object,
                                                group=self.cr_group,
                                                version=self.cr_version,
                                                plural=self.cr_plural,
                                                namespace="default",
                                                body=application_description)

    def delete_spark_application(self, name):
        """
        delete a spark application by its name
        :param name: application name
        :return: bool
        """
        return KubernetesApiHelper.call_k8s_api(self.custom_objects_api.delete_namespaced_custom_object,
                                                group=self.cr_group, version=self.cr_version,
                                                plural=self.cr_plural,
                                                namespace="default", name=name)

    def get_spark_application(self, name, **kwargs) -> KubernetesApiResponse[dict]:
        """
        get spark application info by submission id
        :param name: spark application name
        :return:
        """
        return KubernetesApiHelper.call_k8s_api(self.custom_objects_api.get_namespaced_custom_object,
                                                group=self.cr_group, version=self.cr_version, plural=self.cr_plural,
                                                namespace="default", name=name,
                                                **kwargs)

    def list_spark_application(self, limit, _continue, **kwargs) -> KubernetesApiResponse[dict]:
        """
        list spark applications
        :param limit: limit is a maximum number of responses to return for a list call.
        :param _continue: The continue option should be set when retrieving more results from the server.
        :param kwargs:
        :return:
        """
        return KubernetesApiHelper.call_k8s_api(
            self.custom_objects_api.list_namespaced_custom_object, group=self.cr_group,
            version=self.cr_version, plural=self.cr_plural, namespace="default",
            limit=limit, _continue=_continue, **kwargs)

    def get_spark_pods_by_name(self, name, namespace="default") -> KubernetesApiResponse[V1PodList]:
        """
        get spark pods by job name
        :param name: spark application name
        :param namespace:
        :return: KubernetesApiResponse[V1PodList]
        """
        label_selector = f"sparkoperator.k8s.io/app-name={name}"
        api_response = KubernetesApi().get_pod(namespace=namespace, label_selector=label_selector)
        return api_response
