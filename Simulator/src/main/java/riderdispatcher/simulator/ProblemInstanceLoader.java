package riderdispatcher.simulator;



import com.sun.tools.corba.se.idl.constExpr.Or;
import riderdispatcher.core.*;
import riderdispatcher.utils.Constants;
import riderdispatcher.utils.TimeUtil;
import riderdispatcher.utils.TxtParser;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by bigstone on 27/6/2017.
 */
public class ProblemInstanceLoader {


    private List<Order> loadBaseOrdersByDay(int day) throws UnsupportedEncodingException, FileNotFoundException, InstantiationException, IllegalAccessException {
        List<Order> orders = TxtParser.readFromFile(Order.class, Constants.DATASET_DIR+day+"_"+Constants.ORDER_BASIC_FILE_NAME);
        return orders;
    }

    private List<Driver> loadBaseDriversByCount(int count) throws UnsupportedEncodingException, FileNotFoundException, InstantiationException, IllegalAccessException {
        List<Driver> drivers = TxtParser.readFromFile(Driver.class, Constants.DATASET_DIR+Constants.DRIVER_BASIC_TXT_FILE_NAME);
        return drivers.subList(0, count);
    }


    public ProblemInstance loadProblemInstance(int orderDate, int driverCount) throws FileNotFoundException, InstantiationException, IllegalAccessException, UnsupportedEncodingException, ParseException {
        ProblemInstance instance = new ProblemInstance();
        instance.orders = loadBaseOrdersByDay(orderDate);
        instance.drivers = loadBaseDriversByCount(driverCount);
        instance.orderOracle = loadTaxiDemandOracle(orderDate, Constants.NEW_PREDICTION_FILE_NAME);
        instance.orderReal = loadTaxiDemandOracle(orderDate, Constants.NEW_REAL_DEMAND_FILE_NAME);
        instance.taxiWatchdog = new TaxiDemandSupplyOracle(new ArrayList<>(), 30*60);
        instance.taxiWatchdog.initialTimeFrames();

        long baseTime = TimeUtil.format.parse("2018-06-01 00:00:00").getTime()/1000 + (orderDate-1)*24*60*60;

        //update the time of order to the offset time
        for (Order order: instance.orders){
            order.setStartTime(order.getStartTime() - baseTime);
            order.setEndTime(order.getEndTime() - baseTime);
        }

        return instance;
    }



    public TaxiDemandSupplyOracle loadTaxiDemandOracle(int day, String fileName) throws UnsupportedEncodingException, FileNotFoundException, InstantiationException, IllegalAccessException, ParseException {
        List<ZoneDemandTable> zoneDemandTables = TxtParser.readFromFile(ZoneDemandTable.class, Constants.DATASET_DIR+fileName);
        int frameTime = 30*60;
        int beginTimeOffset = (day-1)*24*60*60;
        int endTimeOffset = day*24*60*60;

        int beginIndex = beginTimeOffset/frameTime;
        int endIndex = endTimeOffset/frameTime;

        TaxiDemandSupplyOracle oracle = new TaxiDemandSupplyOracle(zoneDemandTables.subList(beginIndex, endIndex), frameTime);

        return oracle;
    }

}
