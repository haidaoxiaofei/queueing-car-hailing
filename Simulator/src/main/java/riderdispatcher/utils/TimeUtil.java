package riderdispatcher.utils;

import java.text.DateFormat;
import java.text.SimpleDateFormat;

/**
 * Created by bigstone on 13/6/2017.
 */
public class TimeUtil {
    public static int TIME_FRAME = 60 * 60 * 1000;//60 minutes in millisecond
    public static final int DAY_MILLISECOND = 24 * 60 * 60 * 1000;//24 hours in millisecond
    public static final int OFFSET_MILLISECOND = 540 * 60 * 1000;//9 hours offset in millisecond
    public static DateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
}
