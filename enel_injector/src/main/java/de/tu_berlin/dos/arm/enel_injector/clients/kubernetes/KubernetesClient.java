package de.tu_berlin.dos.arm.enel_injector.clients.kubernetes;

import io.fabric8.kubernetes.api.model.Pod;
import io.fabric8.kubernetes.api.model.PodList;
import io.fabric8.kubernetes.client.Config;
import io.fabric8.kubernetes.client.ConfigBuilder;
import io.fabric8.kubernetes.client.DefaultKubernetesClient;
import io.fabric8.kubernetes.client.dsl.ExecListener;
import io.fabric8.kubernetes.client.dsl.ExecWatch;
import io.fabric8.kubernetes.client.utils.HttpClientUtils;
import okhttp3.OkHttpClient;
import okhttp3.Response;
import org.apache.log4j.Logger;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class KubernetesClient implements AutoCloseable {

    /******************************************************************************
     * INNER CLASSES
     ******************************************************************************/

    static class Listener implements ExecListener {

        private final CompletableFuture<String> data;
        private final ByteArrayOutputStream out;

        public Listener(CompletableFuture<String> data, ByteArrayOutputStream out) {

            this.data = data;
            this.out = out;
        }

        @Override
        public void onOpen(Response response) {

            LOG.info("Reading data... " + response.message());
        }

        @Override
        public void onFailure(Throwable t, Response response) {

            LOG.error(t.getMessage() + " " + response.message());
            data.completeExceptionally(t);
        }

        @Override
        public void onClose(int code, String reason) {

            LOG.info("Exit with: " + code + " and with reason: " + reason);
            data.complete(out.toString());
        }
    }

    /******************************************************************************
     * CLASS VARIABLES
     ******************************************************************************/

    private static final Logger LOG = Logger.getLogger(KubernetesClient.class);

    /******************************************************************************
     * INSTANCE STATE
     ******************************************************************************/

    public final io.fabric8.kubernetes.client.KubernetesClient client;

    /******************************************************************************
     * CONSTRUCTOR(S)
     ******************************************************************************/

    public KubernetesClient() {

        Config config = new ConfigBuilder().build();
        OkHttpClient okHttpClient = HttpClientUtils.createHttpClient(config);
        this.client = new DefaultKubernetesClient(okHttpClient, config);
    }

    /******************************************************************************
     * INSTANCE BEHAVIOURS
     ******************************************************************************/

    public List<Pod> getPodsWithLabel(String namespace, String labelKey, String labelValue) {

        return client.pods().inNamespace(namespace).withLabel(labelKey, labelValue).list().getItems();
    }

    public String execCommandOnPod(String podName, String namespace, String... cmd) throws Exception {

        Pod pod = client.pods().inNamespace(namespace).withName(podName).get();
        LOG.info(
            String.format("Running command: [%s] on pod [%s] in namespace [%s]%n",
            Arrays.toString(cmd), pod.getMetadata().getName(), namespace));

        CompletableFuture<String> data = new CompletableFuture<>();
        try (ExecWatch execWatch = execCmd(pod, data, cmd)) {

            return data.get(10, TimeUnit.SECONDS);
        }
    }

    private ExecWatch execCmd(Pod pod, CompletableFuture<String> data, String... command) {

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        return client.pods()
            .inNamespace(pod.getMetadata().getNamespace())
            .withName(pod.getMetadata().getName())
            .writingOutput(out)
            .writingError(out)
            .usingListener(new Listener(data, out))
            .exec(command);
    }

    @Override
    public void close() {

        this.client.close();
    }
}