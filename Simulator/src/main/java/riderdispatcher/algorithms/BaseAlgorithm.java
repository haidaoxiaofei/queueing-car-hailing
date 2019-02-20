/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package riderdispatcher.algorithms;


import riderdispatcher.core.*;
import riderdispatcher.estimator.IdleTimeEstimator;

import java.util.HashMap;
import java.util.List;

import static riderdispatcher.utils.Constants.lookupLength;

/**
 *
 * @author bigstone
 */
public abstract class BaseAlgorithm {
    abstract public void run(ProblemInstance instance);
    abstract public String getInfo();

    protected ZoneIndex<Order> orderIndex;
    protected ZoneIndex<Driver> driverIndex;

    
    public void buildIndex(ProblemInstance instance){
        orderIndex = new ZoneIndex<>(instance.orders);
        driverIndex = new ZoneIndex<>(instance.drivers);
    }

    public Order findBestOrder(List<Order> zoneOrders, HashMap<Integer, Double> zoneIdleTimeMap, TaxiDemandSupplyOracle orderOracle, TaxiDemandSupplyOracle taxiWatchdog, long currentTimeOffset){
        double minIdleRatio = Double.MAX_VALUE;
        Order selectedOrder = null;
        for (int j = 0; j < zoneOrders.size(); j++) {
            double idleRatio = estimateZoneIdleRatio(zoneOrders.get(j), zoneIdleTimeMap, orderOracle, taxiWatchdog, currentTimeOffset);

            if (idleRatio <= minIdleRatio){
                minIdleRatio = idleRatio;
                selectedOrder = zoneOrders.get(j);
            }
        }
        return selectedOrder;
    }

    public double estimateZoneIdleRatio(Order order, HashMap<Integer, Double> zoneIdleTimeMap, TaxiDemandSupplyOracle orderOracle, TaxiDemandSupplyOracle taxiWatchdog, long currentTimeOffset){
        double idleTime;
        if (zoneIdleTimeMap.containsKey(order.getEndZoneID())){
            idleTime = zoneIdleTimeMap.get(order.getEndZoneID());
        } else {

            double mu = orderOracle.queryRate(currentTimeOffset, currentTimeOffset+lookupLength, order.getEndZoneID()) * 60;
            double lambda = ((taxiWatchdog.queryDemand(currentTimeOffset, currentTimeOffset+lookupLength, order.getEndZoneID()) + 1)/(lookupLength)) * 60;
            int maxDriverCount = (int)Math.ceil(10/lambda);

            idleTime = IdleTimeEstimator.estimateIdleTime(lambda, mu, maxDriverCount) * 60;
            zoneIdleTimeMap.put(order.getEndZoneID(), idleTime);
        }
        double idleRatio = idleTime/(idleTime + order.getEndTime() - order.getStartTime());
        return idleRatio;
    }
}
