package riderdispatcher.simulator;

import riderdispatcher.core.Driver;
import riderdispatcher.core.Grid;
import riderdispatcher.core.Order;
import riderdispatcher.core.RoadNet;
import riderdispatcher.estimator.Estimator;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Env {

    private List<Driver> servingDrivers;
    private Map<Integer, Grid> gridMap;
    private Estimator estimator;

    private int systemTime;
    private static int timeSlice = 1;
    private static int threshold = 24;

    public Env(EnvConf conf, Estimator estimator){
        this.estimator = estimator;

        for(int y = 0; y < conf.gridHeightNum; ++y){
            int ystart = y * conf.gridSize;
            int yend = Math.min((y + 1) * conf.gridSize, conf.maxLat);
            for(int x = 0; x < conf.gridWidthNum; ++x){
                int xstart = x * conf.gridSize;
                int xend = Math.min((x + 1) * conf.gridSize, conf.maxLng);
                int gridid = y * conf.gridWidthNum + x;
                Grid g = new Grid( gridid, ystart, yend, xstart, xend, estimator);
                gridMap.put(gridid, g);
            }
        }
    }

    /**
     * init for each grid
     * 1. driver info
     * 2. order info
    */
    public Status reset(){
        RoadNet.get().reset();
        servingDrivers.clear();
        systemTime = 0;
        List<GridRequest> cityGrids = new ArrayList<GridRequest>();
        for(Grid grid : gridMap.values()){
            grid.reset();
            grid.generateNewOrder(systemTime);
            grid.generateNewDriver(systemTime);
            cityGrids.add(grid.genRequest());
        }

        return new Status(cityGrids);
    }

    /**
     * assign the order to the target driver:
     *
     * calculate after this time interval:
     * 1. checking each serving driver, whether the service is over. if so, put the driver into the current grid
     * 2. generate new order
     * @return
     */
    public Status next(Action action){

        for(GridResponse response : action.getResponse()){
            int gridid = response.getGridid();
            Grid grid = gridMap.get(gridid);
            List<Driver> drivers = grid.update(response.getOdPairList(), systemTime);
            servingDrivers.addAll(drivers);
        }

        systemTime += timeSlice;
        if(systemTime > threshold){
            return null;
        }
        // step 1
        List<Driver> nextTimeServingDrivers = new ArrayList<Driver>();
        for(Driver d : servingDrivers){
            Order o = d.getServingOrder();
            if(o.isExpired(systemTime)){
                // put driver into the current grid's driver list
                int endGridid = o.getEndZoneID();
                Grid endGrid = gridMap.get(endGridid);
                endGrid.addFreeDriver(d, systemTime);
            }else{
                nextTimeServingDrivers.add(d);
            }
        }
        servingDrivers = nextTimeServingDrivers;

        // step 2
        List<GridRequest> cityGrids = new ArrayList<GridRequest>();
        for(Grid grid : gridMap.values()){
            grid.generateNewOrder(systemTime);
            grid.generateNewDriver(systemTime);
            cityGrids.add(grid.genRequest());
        }

        return new Status(cityGrids);
    }
}
