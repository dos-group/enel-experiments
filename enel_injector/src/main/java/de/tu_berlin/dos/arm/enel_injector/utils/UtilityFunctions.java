package de.tu_berlin.dos.arm.enel_injector.utils;

import java.util.Random;

public class UtilityFunctions {

    /******************************************************************************
     * CLASS BEHAVIOURS
     ******************************************************************************/

    public static int getRandomNumberInRange(int min, int max) {

        Random r = new Random();
        return r.ints(min, (max + 1)).findFirst().getAsInt();

    }

}
