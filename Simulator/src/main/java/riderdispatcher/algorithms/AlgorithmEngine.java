package riderdispatcher.algorithms;



import riderdispatcher.core.Order;
import riderdispatcher.core.ProblemInstance;
import riderdispatcher.preProcess.DatasetGenerater;
import riderdispatcher.simulator.DataBatchProvider;
import riderdispatcher.simulator.ProblemInstanceLoader;
import riderdispatcher.utils.Constants;
import riderdispatcher.utils.MyLogger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Random;

/**
 * Created by bigstone on 21/6/2017.
 * For schedule algorithms with predefined orders
 */
public class AlgorithmEngine {
    List<BaseAlgorithm> algorithms = new ArrayList<>();
    MyLogger logger;
    ProblemInstance instance;

    private int day = 1;
    private int frameLength = 10;//seconds
    DatasetGenerater generater = new DatasetGenerater();
    ProblemInstanceLoader loader = new ProblemInstanceLoader();

    List<List> algorithmScores = new ArrayList<>();
    List<List> algorithmTimes = new ArrayList<>();

    Random rand = new Random(2);



    public void varyDriverCounts() throws IOException, ParseException, IllegalAccessException, InstantiationException {
        logger = new MyLogger("varyingDriver.txt");
//        varyingDriver("DriversCount:500", 500);
//        varyingDriver("DriversCount:800", 800);
        varyingDriver("DriversCount:1K", 1000);
        varyingDriver("DriversCount:2K", 2*1000);
        varyingDriver("DriversCount:5K", 5*1000);
        varyingDriver("DriversCount:8K", 8*1000);
        varyingDriver("DriversCount:10K", 10*1000);
//        varyingDriver("DriversCount:20K", 20*1000);

    }








    private void loadProblem(String info, int day, int driverCount) throws UnsupportedEncodingException, FileNotFoundException, InstantiationException, ParseException, IllegalAccessException {

        instance = loader.loadProblemInstance(day, driverCount);
        instance.info = info;

    }


    private void varyingDriver(String info, int driverCount) throws IOException, ParseException, IllegalAccessException, InstantiationException {
        loadProblem(info, day, driverCount);
        Constants.lookupLength = 5*60;
        runAlgorithms();
    }

    public void run() throws IOException, ParseException, IllegalAccessException, InstantiationException {
        varyDriverCounts();


    }

    public AlgorithmEngine() throws ParseException {
        this.algorithms.add(new IdleRatioGreedyAlgorithm(false, "IdleRatioGreedyAlgorithm"));
        this.algorithms.add(new IdleRatioGreedyAlgorithm(true, "RealIdleRatioGreedyAlgorithm"));
        this.algorithms.add(new LongTripGreedyAlgorithm());
        this.algorithms.add(new RandomAlgorithm());
        this.algorithms.add(new UpperAlgorithm());


        algorithms.forEach(algorithm -> {
            algorithmScores.add(new ArrayList<>());
            algorithmTimes.add(new ArrayList<>());
        });
    }

    private void runAlgorithms() throws IOException{
        for (BaseAlgorithm algorithm : algorithms) {
            ProblemInstance tmpInstance = new ProblemInstance(instance);

            long millisStart = Calendar.getInstance().getTimeInMillis();
            DataBatchProvider batchProvider = new DataBatchProvider(tmpInstance, frameLength);

            ProblemInstance lastInstance = null;
            ProblemInstance currentInstance;
            while ((currentInstance = batchProvider.fetchCurrentProblemInstance())!= null){
                if (lastInstance != null){
                    long currentTimeOffset = currentInstance.currentTimeOffset;
                    for (Order order: lastInstance.orders){
                        if (!order.isExpired(currentTimeOffset) && !order.isAssigned()){
                            currentInstance.orders.add(order);
                        }

                        if (order.isAssigned()){
                            tmpInstance.completedOrders.add(order);
                        }

                        if (!order.isAssigned() && order.isExpired(currentTimeOffset)){
                            tmpInstance.expiredOrders.add(order);
                        }
                    }
                }
                algorithm.run(currentInstance);

                lastInstance = currentInstance;

            }


            long millisEnd = Calendar.getInstance().getTimeInMillis();
            tmpInstance.info = algorithm.getInfo() + ":" + tmpInstance.info;
            tmpInstance.startRunningMillis = millisStart;
            tmpInstance.endRunningMillis = millisEnd;


//            int index = algorithms.indexOf(algorithm);
//            algorithmTimes.get(index).add((double)millisEnd - millisStart);
//            algorithmScores.get(index).add(tmpInstance.calculateTotalDistance());

            logger.logResult(tmpInstance);
        }
        System.out.println();

    }

}
