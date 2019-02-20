package riderdispatcher.utils;

import riderdispatcher.core.Order;
import riderdispatcher.core.ProblemInstance;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

/**
 * Created by bigstone on 23/6/2017.
 */
public class MyLogger {

    private static PrintWriter pw;

    public MyLogger(String fileName) throws IOException {
        pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(Constants.RESULT_DIR + fileName), "utf-8"));

    }

    public void logResult(ProblemInstance instance) throws IOException {



        String template = "{info},RunningTime:{running},servingTime:{servingTime},assignedOrderCount:{orderCount}";
        template = template.replace("{info}", instance.info);
        template = template.replace("{running}", String.valueOf(((double)instance.endRunningMillis - instance.startRunningMillis)));
        template = template.replace("{servingTime}", String.valueOf(instance.calculateTotalServingTime()));
        template = template.replace("{orderCount}", String.valueOf(instance.calculateTotalAssignedOrderCount()));


        pw.println(template);
        pw.flush();
        System.out.println(template);
    }

}