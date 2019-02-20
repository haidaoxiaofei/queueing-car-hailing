package riderdispatcher.core;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class RoadNet {
    private Properties props = new Properties();

    public int minLng;
    public int maxLng;
    public int minLat;
    public int maxLat;

    public int gridSize;
    public int gridWidthNum;
    public int gridHeightNum;
    public int gridNum;

    public AtomicInteger driverIdGenerator = new AtomicInteger();
    public AtomicInteger orderIdGenerator = new AtomicInteger();

    private static RoadNet singleton;

    public static RoadNet constructRoadNet(String confPath) throws IOException {
        if(singleton == null) {
            singleton = new RoadNet(confPath);
        }
        return singleton;
    }

    public static RoadNet get(){
        return singleton;
    }

    private RoadNet(String confPath) throws IOException {
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
        gridNum = gridHeightNum * gridWidthNum;
    }

    private int[] genGridRange(int gridid){
        int rowIdx = gridid / gridWidthNum;
        int colIdx = gridid % gridWidthNum;
        int ystart = rowIdx * gridSize;
        int yend = Math.min((rowIdx + 1) * gridSize, maxLat);
        int xstart = colIdx * gridSize;
        int xend = Math.min((colIdx + 1) * gridSize, maxLng);
        return new int[]{ystart, yend, xstart, xend};
    }

    private int randomGridid(){
        Random rPos = new Random();
        return rPos.nextInt(gridNum);
    }

    private Point randomPos(int gridid){
        Random r = new Random();
        int[] range = genGridRange(gridid);
        int yPos = (int)(r.nextFloat() * (range[1] - range[0]) + range[0]);
        int xPos = (int)(r.nextFloat() * (range[3] - range[2]) + range[2]);
        return new Point(yPos, xPos);
    }

    public synchronized Driver randomDriver(int gridid){
        Driver d = new Driver();
        d.setCurrentZoneID(gridid);
        d.setCurPos(randomPos(gridid));
        d.setId(driverIdGenerator.incrementAndGet());
        return d;
    }

    public synchronized  Order randomOrder(int gridid){
        Order o = new Order();
        o.setStartZoneID(gridid);
        o.setStartPoint(randomPos(gridid));
        int endGridid = randomGridid();
        o.setEndZoneID(endGridid);
        o.setEndPoint(randomPos(endGridid));
        o.setId(orderIdGenerator.incrementAndGet());
        return o;
    }

    private int[] getGridIdx(int gridid){
        int rowIdx = gridid / gridWidthNum;
        int colIdx = gridid % gridWidthNum;
        return new int[]{rowIdx, colIdx};
    }

    private double eta(Order order){
        int sid = order.getStartZoneID();
        int[] sIdx = getGridIdx(sid);
        int eid = order.getEndZoneID();
        int[] eIdx = getGridIdx(eid);
        int dist = Math.abs(sIdx[0] - eIdx[0]) + Math.abs(sIdx[1] - eIdx[1]);
        return order.getStartTime() + dist;
    }

    public void acceptOrder(Driver driver, Order order, int startTime){
        order.setStartTime(startTime);
        order.setEndTime((long)eta(order));
        order.setCost(order.getEndTime() - order.getStartTime());
        driver.setServingOrder(order);
    }

    public synchronized void reset() {
        driverIdGenerator.set(0);
        orderIdGenerator.set(0);
    }
}
