package de.tu_berlin.dos.arm.enel_injector.clients;

import de.tu_berlin.dos.arm.enel_injector.clients.kubernetes.KubernetesClient;
import de.tu_berlin.dos.arm.enel_injector.utils.UtilityFunctions;
import io.fabric8.kubernetes.api.model.Pod;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ClientsManager {

    /******************************************************************************
     * CLASS VARIABLES
     ******************************************************************************/

    private static final Logger LOG = Logger.getLogger(ClientsManager.class);

    /******************************************************************************
     * INSTANCE STATE
     ******************************************************************************/

    public final StopWatch stopWatch = new StopWatch();
    public final KubernetesClient k8sClient;
    public final String namespace;

    /******************************************************************************
     * CONSTRUCTOR(S)
     ******************************************************************************/

    public ClientsManager(String namespace) {

        this.k8sClient = new KubernetesClient();
        this.namespace = namespace;
    }

    /******************************************************************************
     * INSTANCE BEHAVIOUR
     ******************************************************************************/

    public void injectRandomFailure(String labelKey, String labelValue) throws Exception {

        List<Pod> pods = this.k8sClient.getPodsWithLabel(namespace, labelKey, labelValue);
        if (0 < pods.size()) {

            stopWatch.start();
            while (stopWatch.getTime(TimeUnit.SECONDS) < 10) {

                int index = UtilityFunctions.getRandomNumberInRange(0, pods.size());
                Pod pod = pods.get(index);
                LOG.info(pod.getStatus().getPhase());
                if ("Running".equalsIgnoreCase(pod.getStatus().getPhase())) {

                    String podName = pod.getMetadata().getName();
                    this.k8sClient.execCommandOnPod(podName, this.namespace, "sh", "-c", "kill 1");
                    break;
                }
                else LOG.info(String.format("Pod status phase was %s", pod.getStatus().getPhase()));
            }
            this.stopWatch.reset();
        }
        else LOG.info(String.format("No pods found with label <%s:%s> in namespace %s", labelKey, labelValue, namespace));
    }
}
