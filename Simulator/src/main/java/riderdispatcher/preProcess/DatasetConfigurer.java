package riderdispatcher.preProcess;

import riderdispatcher.core.*;
import riderdispatcher.utils.Constants;
import riderdispatcher.utils.TimeUtil;
import riderdispatcher.utils.TxtParser;


import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.text.ParseException;
import java.util.*;

/**
 * Created by bigstone on 14/6/2017.
 */
public class DatasetConfigurer {

    public List<Order> configureOrders(List<OrderRecord> records) throws IOException {
        int orderCount = 0;
        List<Order> orders = new ArrayList<>();
        for (OrderRecord record: records){
            Order order = new Order(record, orderCount++);
            orders.add(order);
        }
        return orders;
    }

    public static void configDemandReal() throws IOException, InstantiationException, IllegalAccessException {
        List<ZoneDemandTable> demandStatisticTables = TxtParser.readFromFile(ZoneDemandTable.class, Constants.DATASET_DIR+Constants.REAL_DEMAND_FILE_NAME);

        TxtParser.writeToFile(demandStatisticTables, Constants.DATASET_DIR+Constants.NEW_REAL_DEMAND_FILE_NAME);

    }


    public static void summaryOrderRecords() throws IOException, InstantiationException, IllegalAccessException {
        ZoneDemandTable demandStatisticTable = new ZoneDemandTable();
        List<OrderRecord> records = TxtParser.readFromFile(OrderRecord.class, Constants.DATASET_DIR+Constants.CLEAN_ORDER_RECORD_FILE_NAME);

        for (int i = 0; i < records.size(); i++) {
            demandStatisticTable.addDemand(records.get(i).pickZoneID, 1);
        }
        List<ZoneDemandTable> demandTables = new ArrayList<>();
        demandTables.add(demandStatisticTable);
        TxtParser.writeToFile(demandTables, Constants.DATASET_DIR+Constants.DEMAND_DISTRIBUTION_FILE_NAME);
    }

    public static void summaryDrivers() throws IOException, InstantiationException, IllegalAccessException {
        ZoneDemandTable demandStatisticTable = new ZoneDemandTable();
        List<Driver> drivers = TxtParser.readFromFile(Driver.class, Constants.DATASET_DIR+Constants.DRIVER_BASIC_TXT_FILE_NAME);

        for (int i = 0; i < drivers.size(); i++) {
            demandStatisticTable.addDemand(drivers.get(i).getCurrentZoneID(), 1);
        }
        List<ZoneDemandTable> demandTables = new ArrayList<>();
        demandTables.add(demandStatisticTable);
        TxtParser.writeToFile(demandTables, Constants.DATASET_DIR+Constants.DRIVER_DISTRIBUTION_FILE_NAME);
    }

    public static void cleanOrderRecords() throws IOException, InstantiationException, IllegalAccessException, ParseException {
        List<OrderRecord> yellowRecords = TxtParser.readFromFile(OrderRecord.class, Constants.DATASET_DIR+Constants.YELLOW_ORDER_FILE_NAME);
        List<OrderRecord> greenRecords = TxtParser.readFromFile(OrderRecord.class, Constants.DATASET_DIR+Constants.GREEN_ORDER_FILE_NAME);
        List<OrderRecord> totalRecords = new ArrayList<>();
        totalRecords.addAll(yellowRecords);
        totalRecords.addAll(greenRecords);

        long minTimestamp = TimeUtil.format.parse("2018-06-01 00:00:00").getTime()/1000;
        long maxTimestamp = TimeUtil.format.parse("2018-07-01 00:00:00").getTime()/1000;

        List<OrderRecord> cleanRecords = new ArrayList<>();

        for (OrderRecord record: totalRecords){
            if (record.pickZoneID <= 263
                    && record.dropZoneID <= 263
                    && record.pickTime > minTimestamp
                    && record.pickTime < maxTimestamp
                    && record.dropTime - record.pickTime >= 3*60
                    && record.dropTime - record.pickTime <= 24*60*60){//normal trips should be at least 3 minutes
                cleanRecords.add(record);
            }
        }

        Collections.sort(cleanRecords, ( r1,  r2) -> (int)Math.ceil(r1.pickTime - r2.pickTime));

        TxtParser.writeToFile(cleanRecords, Constants.DATASET_DIR+Constants.CLEAN_ORDER_RECORD_FILE_NAME);
    }

    public static void main(String[] args) throws IOException, InstantiationException, IllegalAccessException, ParseException {
//        List<OrderRecord> records = TxtParser.readFromFile(OrderRecord.class, Constants.DATASET_DIR+Constants.ORDER_SAMPLE_FILE_NAME);
//
//        DatasetConfigurer datasetConfigurer = new DatasetConfigurer();
//        List<Order> orders = datasetConfigurer.configureOrders(records);
//        TxtParser.writeToFile(orders, Constants.DATASET_DIR+ Constants.ORDER_BASIC_FILE_NAME);
//


//        cleanOrderRecords();
//        summaryOrderRecords();
//        summaryDrivers();

        configDemandReal();

        System.out.println("Finish Configuring Dataset...");
    }
}
