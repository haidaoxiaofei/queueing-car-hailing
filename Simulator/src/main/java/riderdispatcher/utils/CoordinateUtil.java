package riderdispatcher.utils;

/**
 * Created by bigstone on 21/6/2017.
 */
public class CoordinateUtil {
    public static final double MAX_LATITUDE = 35.867150422391695;
    public static final double MIN_LATITUDE = 35.510184690694565;
    public static final double MAX_LONGITUDE = 139.91259321570396;
    public static final double MIN_LONGITUDE = 139.4708776473999;

    public static double convertLatitude(double latitude){
        return (latitude - MIN_LATITUDE) / (MAX_LATITUDE - MIN_LATITUDE);
    }

    public static double convertLongitude(double longitude){
        return (longitude - MIN_LONGITUDE) / (MAX_LONGITUDE - MIN_LONGITUDE);
    }
}
