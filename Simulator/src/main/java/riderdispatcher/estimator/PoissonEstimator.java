package riderdispatcher.estimator;

import riderdispatcher.core.Driver;
import riderdispatcher.core.RoadNet;
import riderdispatcher.simulator.EnvConf;
import riderdispatcher.core.Order;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PoissonEstimator implements Estimator{
    public static int getPoisson(double lambda) {
        double L = Math.exp(-lambda);
        double p = 1.0;
        int k = 0;

        do {
            k++;
            p *= Math.random();
        } while (p > L);

        return k - 1;
    }

    private List<Double> newDriverParams;
    private List<Double> newOrderParams;

    private static final int waiting_mean = 2;

    private void loadVector(String file, List<Double> vec){
        try (BufferedReader br = new BufferedReader(new FileReader(file))){
            String line;
            while((line = br.readLine()) != null){
                String[] sps = line.split("");
                for(String sp : sps){
                    double lambda = Double.parseDouble(sp);
                    vec.add(lambda);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public PoissonEstimator(EnvConf conf, String driverFile, String orderFile){
        loadVector(driverFile, newDriverParams);
        loadVector(orderFile, newOrderParams);
    }

    /**
     * 2 steps:
     * 1. generate driver numbers
     * 2. generate driver detail: start position will be random
     * @param gridid
     * @param systemTime
     * @return
     */
    @Override
    public List<Driver> genRandomDrivers(int gridid, int systemTime) {
        Double mean = newDriverParams.get(gridid);
        int realNum = getPoisson(mean);
        List<Driver> driverList = new ArrayList<>();
        for(int i = 0; i < realNum; ++i){
            Driver d = RoadNet.get().randomDriver(gridid);
            driverList.add(d);
        }
        return driverList;
    }

    @Override
    public List<Order> genRandomOrders(int gridid, int systemTime) {
        Double mean = newOrderParams.get(gridid);
        int realNum = getPoisson(mean);
        List<Order> orderList = new ArrayList<>();
        for(int i = 0; i < realNum; ++i){
            Order o = RoadNet.get().randomOrder(gridid);
            // TODO:
            o.setMaxWaitTime(waiting_mean);
            orderList.add(o);
        }
        return orderList;
    }

    public boolean isDriverQuit(Driver d, int systemTime){
        Random r = new Random();
        return r.nextFloat() < 0.03;
    }
}
