package riderdispatcher.algorithms;

import riderdispatcher.core.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class IdleRatioGreedyAlgorithm extends BaseAlgorithm {
    private String info;

    private boolean isRealDemand;

    public IdleRatioGreedyAlgorithm(boolean isRealDemand, String info){
        this.isRealDemand = isRealDemand;
        this.info = info;
    }

    @Override
    public void run(ProblemInstance instance) {
        if (instance.orders.isEmpty()){
            return;
        }
        buildIndex(instance);
        long currentTimeOffset = instance.currentTimeOffset;
        HashMap<Integer, Double> zoneIdleTimeMap = new HashMap<>();
        TaxiDemandSupplyOracle oracle;

        if (isRealDemand){
            oracle = instance.orderReal;
        } else {
            oracle = instance.orderOracle;
        }
        for (int i = 0; i < ZoneDemandTable.TOTAL_ZONE_COUNT; i++) {

            List<Order> zoneOrders = orderIndex.queryZoneObjects(i+1);
            List<Driver> zoneDrivers = driverIndex.queryZoneObjects(i+1);
            Order selectedOrder;
            while (!zoneDrivers.isEmpty() && !zoneOrders.isEmpty()){
                List<Driver> assignedDrivers = new ArrayList<>();
                for (Driver driver: zoneDrivers){
                    selectedOrder = findBestOrder(zoneOrders,  zoneIdleTimeMap, oracle, instance.taxiWatchdog, currentTimeOffset );

                    if (driver != null && selectedOrder != null){
                        driver.serveOrder(selectedOrder, currentTimeOffset);
                        zoneOrders.remove(selectedOrder);
                        assignedDrivers.add(driver);
                        instance.taxiWatchdog.addTimeRecord(selectedOrder.getEndZoneID(), currentTimeOffset + selectedOrder.getEndTime() - selectedOrder.getStartTime(), 1);
                        zoneIdleTimeMap.remove(selectedOrder.getEndZoneID());
                    }
                }
                zoneDrivers.removeAll(assignedDrivers);
            }
        }
    }




    @Override
    public String getInfo() {
        return info;
    }
}
