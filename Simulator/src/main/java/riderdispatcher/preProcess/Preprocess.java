package riderdispatcher.preProcess;

import java.io.IOException;
import java.text.ParseException;

/**
 * Created by bigstone on 21/6/2017.
 */
public class Preprocess {
    public static void main(String[] args) throws IOException, IllegalAccessException, InstantiationException, ParseException {
        DatasetConfigurer.main(null);
        DatasetGenerater.main(null);
    }
}
