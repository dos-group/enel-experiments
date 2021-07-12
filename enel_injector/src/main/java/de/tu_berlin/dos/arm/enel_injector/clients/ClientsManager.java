package de.tu_berlin.dos.arm.enel_injector.clients;

import de.tu_berlin.dos.arm.enel_injector.clients.kubernetes.KubernetesClient;
import de.tu_berlin.dos.arm.enel_injector.utils.UtilityFunctions;
import io.fabric8.kubernetes.api.model.Pod;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
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
    public final int minPods;
    public final String namespace;

    /******************************************************************************
     * CONSTRUCTOR(S)
     ******************************************************************************/

    public ClientsManager(int minPods, String namespace) {

        this.k8sClient = new KubernetesClient();
        this.minPods = minPods;
        this.namespace = namespace;
    }

    /******************************************************************************
     * INSTANCE BEHAVIOUR
     ******************************************************************************/

    public void injectRandomFailure(String labelKey, String labelValue) {

        try {
            List<Pod> pods = this.k8sClient.getPodsWithLabel(namespace, labelKey, labelValue);
            LOG.info(String.format(
                "Number of pods detected with label <%s:%s> in namespace %s is %d",
                labelKey, labelValue, namespace, pods.size()));
            if (this.minPods < pods.size()) {

                stopWatch.start();
                while (stopWatch.getTime(TimeUnit.SECONDS) < 10) {

                    // determine if all pods are in running phase
                    List<String> validPhases = Arrays.asList("Running", "ContainerCreating");
                    boolean allAreRunning = true;
                    for (Pod pod : pods) {

                        if (!validPhases.contains(pod.getStatus().getPhase())) allAreRunning = false;
                    }
                    // if they are, select random pod and kill it
                    if (allAreRunning) {

                        int index = UtilityFunctions.getRandomNumberInRange(0, pods.size() - 1);
                        String podName = pods.get(index).getMetadata().getName();
                        this.k8sClient.execCommandOnPod(podName, this.namespace, "sh", "-c", "kill 1");
                        break;
                    }
                    else LOG.info("Not all pods status were in Running or ContainerCreating phases");
                    new CountDownLatch(1).await(100, TimeUnit.MILLISECONDS);
                }
                this.stopWatch.reset();
            }
            else LOG.info(String.format("No pods found with label <%s:%s> in namespace %s", labelKey, labelValue, namespace));
        }
        catch (Exception e) {

            e.printStackTrace();
        }
    }
}
