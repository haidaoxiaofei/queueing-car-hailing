package riderdispatcher.algorithms;

import riderdispatcher.core.Driver;
import riderdispatcher.core.Order;
import riderdispatcher.core.ProblemInstance;
import riderdispatcher.core.ZoneDemandTable;

import java.util.List;
import java.util.Random;

public class RandomAlgorithm extends BaseAlgorithm{
    public static final String info = "RandomAlgorithm";

    @Override
    public void run(ProblemInstance instance) {
        if (instance.orders.isEmpty()){
            return;
        }
        buildIndex(instance);
        for (int i = 0; i < ZoneDemandTable.TOTAL_ZONE_COUNT; i++) {

            List<Order> zoneOrders = orderIndex.queryZoneObjects(i+1);
            List<Driver> zoneDrivers = driverIndex.queryZoneObjects(i+1);
            Random rand = new Random(1);
            for (Driver driver: zoneDrivers){
                if (zoneOrders.isEmpty()){
                    break;
                }
                int orderIndex = rand.nextInt(zoneOrders.size());
                driver.serveOrder(zoneOrders.get(orderIndex), instance.currentTimeOffset);
                zoneOrders.remove(orderIndex);
            }
        }
    }

    @Override
    public String getInfo() {
        return info;
    }
}
