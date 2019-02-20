package riderdispatcher.simulator;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

public class EnvConf {
    private Properties props = new Properties();

    public int minLng;
    public int maxLng;
    public int minLat;
    public int maxLat;

    public int gridSize;
    public int gridWidthNum;
    public int gridHeightNum;

    public EnvConf(String confPath) throws IOException {
        FileInputStream fis = new FileInputStream(confPath);
        props.load(fis);

        minLng = Integer.parseInt(props.getProperty("minLng"));
        maxLng = Integer.parseInt(props.getProperty("maxLng"));
        minLat = Integer.parseInt(props.getProperty("minLat"));
        maxLat = Integer.parseInt(props.getProperty("maxLat"));

        gridSize = Integer.parseInt(props.getProperty("gridSIze"));

        assert gridSize > 0;
        gridWidthNum = (maxLng - minLng) / gridSize;
        if( (maxLng - minLng) % gridSize != 0 ){
            gridWidthNum += 1;
        }
        gridHeightNum = (maxLat - minLat) / gridSize;
        if( (maxLat - minLat) % gridSize != 0 ){
            gridHeightNum += 1;
        }
    }


}
