package de.tu_berlin.dos.arm.enel_injector;

import de.tu_berlin.dos.arm.enel_injector.clients.ClientsManager;
import de.tu_berlin.dos.arm.enel_injector.utils.FileReader;
import de.tu_berlin.dos.arm.enel_injector.utils.UtilityFunctions;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.log4j.Logger;

import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class Run {

    /******************************************************************************
     * CLASS VARIABLES
     ******************************************************************************/

    private static final Logger LOG = Logger.getLogger(Run.class);

    /******************************************************************************
     * MAIN
     ******************************************************************************/

    public static void main(String[] args) throws Exception {

        // get properties file
        Properties props = FileReader.GET.read("enel_injector.properties", Properties.class);

        // get properties for failure experiments
        int interval = Integer.parseInt(props.getProperty("general.interval"));
        String namespace = props.getProperty("kubernetes.namespace");
        String labelKey = props.getProperty("kubernetes.labelKey");
        String labelValue = props.getProperty("kubernetes.labelValue");

        ClientsManager client = new ClientsManager(namespace);

        // execute failure loop based on interval
        final StopWatch stopWatch = new StopWatch();
        while (true) {

            // wait random amount of seconds to inject failure
            int waitTime = UtilityFunctions.getRandomNumberInRange(0, interval);
            LOG.info(String.format("Waiting %s seconds to inject failure", waitTime));
            stopWatch.start();
            long current = stopWatch.getTime(TimeUnit.SECONDS);
            while (current < waitTime) {

                current = stopWatch.getTime(TimeUnit.SECONDS);
                new CountDownLatch(1).await(100, TimeUnit.MILLISECONDS);
            }

            // inject failure
            LOG.info("Injecting Failure");
            client.injectRandomFailure(labelKey, labelValue);

            // wait until end of interval
            LOG.info(String.format("Waiting %s seconds until end of interval", interval - waitTime));
            while (current < interval) {

                current = stopWatch.getTime(TimeUnit.SECONDS);
                new CountDownLatch(1).await(100, TimeUnit.MILLISECONDS);
            }
            stopWatch.reset();
        }
    }
}
