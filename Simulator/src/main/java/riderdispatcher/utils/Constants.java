/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package riderdispatcher.utils;

/**
 *
 * @author bigstone
 */
public class Constants {

    public static final String DATASET_DIR = "/Users/bigstone/workspace/java/ridedispatcher_ICDE/dataset/raw_data/";
    public static final String RESULT_DIR = "/Users/bigstone/workspace/MATLAB/spatialQueueing/results/";


    public static final String REAL_DEMAND_FILE_NAME = "deepst_gc.txt_real";
    public static final String NEW_REAL_DEMAND_FILE_NAME = "new_deepst_gc.txt_real";
    public static final String NEW_PREDICTION_FILE_NAME = "new_deepst_gc.txt";
    public static final String YELLOW_ORDER_FILE_NAME = "yellow_tripdata_2018-06.csv";
    public static final String GREEN_ORDER_FILE_NAME = "green_tripdata_2018-06.csv";
    public static final String CLEAN_ORDER_RECORD_FILE_NAME = "clean_tripdata_2018-06.csv";
    public static final String DEMAND_DISTRIBUTION_FILE_NAME = "demand_distribution_2018-06.csv";
    public static final String DRIVER_DISTRIBUTION_FILE_NAME = "driver_distribution_2018-06.csv";


    public static final String ORDER_BASIC_FILE_NAME = "orders_basic.txt";
    public static final String DRIVER_BASIC_TXT_FILE_NAME = "drivers_basic.txt";

    public static long lookupLength = 5*60;
}
