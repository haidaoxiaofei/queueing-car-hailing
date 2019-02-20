package riderdispatcher.preProcess;

/**
 * Created by bigstone on 21/6/2017.
 */



import riderdispatcher.core.*;
import riderdispatcher.utils.Constants;
import riderdispatcher.utils.TimeUtil;
import riderdispatcher.utils.TxtParser;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 *
 * @author bigstone
 */
public class DatasetGenerater {
    public static int MAX_DRIVER_COUND = 300*1000;//for exp dir
    long minTimestamp = TimeUtil.format.parse("2018-06-01 00:00:00").getTime()/1000;

    private static double lambda = 1;//1 rider per minute



    public DatasetGenerater() throws ParseException {
    }

    public List<Driver> generateRealBaseDrivers() throws UnsupportedEncodingException, FileNotFoundException, InstantiationException, IllegalAccessException {
        List<ZoneDemandTable> zoneDemandTables = TxtParser.readFromFile(ZoneDemandTable.class,Constants.DATASET_DIR+Constants.DEMAND_DISTRIBUTION_FILE_NAME);
        ZoneDemandTable zoneDemandTable = zoneDemandTables.get(0);

        List<Double> zoneDemandDistribution = zoneDemandTable.getDemandDistribution();
        Random rand = new Random(1);

        List<Driver> drivers = new ArrayList<>();

        for (int i = 0; i < MAX_DRIVER_COUND; i++) {
            int nextZoneId = rollForNextZone(zoneDemandDistribution, rand);
            Driver driver = new Driver();
            driver.setNextFreeTimeOffset(0);
            driver.setCurrentZoneID(nextZoneId);
            drivers.add(driver);
        }
        return drivers;
    }


    public List<Order> generateRealBaseOrders() throws IOException, InstantiationException, IllegalAccessException {
        List<OrderRecord> records = TxtParser.readFromFile(OrderRecord.class, Constants.DATASET_DIR+Constants.CLEAN_ORDER_RECORD_FILE_NAME);
        Random rand = new Random(1);
        int orderCount = 0;

        List<Order> orders = new ArrayList<>();
        for (OrderRecord record: records){
            Order order = new Order(record, orderCount++);
            order.setMaxWaitTime(nextRandomWaitingTime(120, rand));
            orders.add(order);
        }
        return orders;
    }

    public List<List<Order>> groupOrdersByDay(List<Order> orders){
        List<List<Order>> ordersByDay = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            ordersByDay.add(new ArrayList());
        }

        for (Order order: orders){
            int offsetDay = (int)(order.getStartTime() - minTimestamp) / (24*60*60);
            if (offsetDay > 30) {
                System.out.println("Bad!");
            }
            ordersByDay.get(offsetDay).add(order);
        }

        return ordersByDay;
    }

    private int rollForNextZone(List<Double> zoneDemandDistribution, Random rand){
        float nextRoll = rand.nextFloat();
        int selectedIndex = -1;
        float accumulatedFloat = 0;

        for (int index = 0; index<zoneDemandDistribution.size(); index++){
            accumulatedFloat += zoneDemandDistribution.get(index);
            if (accumulatedFloat > nextRoll){
                selectedIndex = index;
                break;
            }
        }

        return selectedIndex + 1;
    }

    private double nextRandomWaitingTime(double baseWaitingTime, Random rand){
        double accumulatedP = rand.nextDouble();
        double waitingTime = baseWaitingTime + Math.log(1-accumulatedP)/(-lambda) * 60;
        return waitingTime;
    }


    //lambda : number of quit riders per minute
    public void regenerateMaxWaitingTimes(ProblemInstance instance, double baseWaitingTime, Random rand){
        for (int i = 0; i < instance.orders.size(); i++) {
            Order currentOrder = instance.orders.get(i);
            currentOrder.setMaxWaitTime(nextRandomWaitingTime(baseWaitingTime, rand));
        }
    }

    public static void main(String[] args) throws IOException, ParseException, IllegalAccessException, InstantiationException {

        DatasetGenerater generater = new DatasetGenerater();

        List<Driver> drivers = generater.generateRealBaseDrivers();
        List<Order> orders = generater.generateRealBaseOrders();
        List<List<Order>> ordersByDay = generater.groupOrdersByDay(orders);

        TxtParser.writeToFile(drivers, Constants.DATASET_DIR+Constants.DRIVER_BASIC_TXT_FILE_NAME);
        for (int i = 0; i < 30; i++) {
            TxtParser.writeToFile(ordersByDay.get(i), Constants.DATASET_DIR+(i+1)+"_"+Constants.ORDER_BASIC_FILE_NAME);
        }

    }

}
