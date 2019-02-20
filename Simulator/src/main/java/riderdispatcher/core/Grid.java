package riderdispatcher.core;

import riderdispatcher.estimator.Estimator;
import riderdispatcher.simulator.GridRequest;
import riderdispatcher.simulator.GridResponse;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Grid {
    private int gridid;
    private int ystart;
    private int yend;
    private int xstart;
    private int xend;
    private Estimator estimator;

//    private List<Driver> driverList;

    private Map<Integer, Driver> freeDrivers = new HashMap<>();
    private Map<Integer, Order> freeOrders = new HashMap<>();

    public Grid(int gridid, int ystart, int yend, int xstart, int xend, Estimator estimator){
        this.gridid = gridid;
        this.ystart = ystart;
        this.yend = yend;
        this.xstart = xstart;
        this.xend = xend;
        this.estimator = estimator;
    }

    /**
     * currently we only support single thread generate new order
     * @param systemTime
     */
    public void generateNewOrder(int systemTime){
        for(Driver newDriver : estimator.genRandomDrivers(gridid, systemTime)){
            freeDrivers.put(newDriver.getId(), newDriver);
        }
    }

    /**
     * currently we only support single thread generate new driver
     * @param systemTime
     */
    public void generateNewDriver(int systemTime){
        for(Order newOrder : estimator.genRandomOrders(gridid, systemTime)){
            freeOrders.put(newOrder.getId(), newOrder);
        }
    }

    public List<Driver> update(List<GridResponse.ODPair> odPairList, int currentTime) {
        List<Driver> servingDrivers = new ArrayList<>();
        for(GridResponse.ODPair pair : odPairList){
            Driver driver = freeDrivers.get(pair.driverid);
            Order order = freeOrders.get(pair.orderid);
            driver.serveOrder(order, currentTime);
            RoadNet.get().acceptOrder(driver, order, currentTime);

            freeDrivers.remove(pair.driverid);
            freeOrders.remove(pair.orderid);
            servingDrivers.add(driver);
        }
        return servingDrivers;
    }

    public GridRequest genRequest() {
        List<Driver> driverList = new ArrayList<>(freeDrivers.values());
        List<Order> orderList = new ArrayList<>(freeOrders.values());
        return new GridRequest(driverList, orderList);
    }

    public void addFreeDriver(Driver driver, int currentTime){
        if(!estimator.isDriverQuit(driver, currentTime)){
            freeDrivers.put(driver.getId(), driver);
        }
    }

    public int getGridid() {
        return gridid;
    }

    public void setGridid(int gridid) {
        this.gridid = gridid;
    }

    public void reset() {
        freeOrders.clear();
        freeDrivers.clear();
    }
}
