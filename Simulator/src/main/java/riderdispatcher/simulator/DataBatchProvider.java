package riderdispatcher.simulator;

import riderdispatcher.core.Driver;
import riderdispatcher.core.Order;
import riderdispatcher.core.ProblemInstance;

import java.util.ArrayList;
import java.util.List;

public class DataBatchProvider {

    private int frameLength;
    private long currentTimeOffset = 0;
    private int currentOrderIndex = 0;


    ProblemInstance instance;


    public DataBatchProvider(ProblemInstance baseInstance, int frameLength) {
        this.frameLength = frameLength;
        this.instance = baseInstance;
    }



    public void slideToNextRound(){
        this.currentTimeOffset += frameLength;
    }

    public List<Driver> fetchCurrentRoundDrivers(){
        List<Driver> drivers = new ArrayList();

        for (Driver driver: instance.drivers){
            if (driver.getNextFreeTimeOffset() <= currentTimeOffset){
                drivers.add(driver);
            }
        }


        return drivers;
    }

    public List<Order> fetchCurrentRoundOrders(){
        List<Order> orders = new ArrayList();

        //orders have been sorted by the startingTime
        for (Order order: instance.orders.subList(currentOrderIndex, instance.orders.size())){
            if (order.getStartTime() <= currentTimeOffset){
                orders.add(order);
                currentOrderIndex++;
            }
        }

        return orders;
    }

    public ProblemInstance fetchCurrentProblemInstance(){
        if (currentTimeOffset > instance.orders.get(instance.orders.size() - 1).getStartTime()){
            return null;
        }
        ProblemInstance instance = new ProblemInstance();
        instance.drivers = fetchCurrentRoundDrivers();
        instance.orders = fetchCurrentRoundOrders();
        instance.currentTimeOffset = currentTimeOffset;
        instance.taxiWatchdog = this.instance.taxiWatchdog;
        instance.orderOracle = this.instance.orderOracle;
        instance.orderReal = this.instance.orderReal;
        slideToNextRound();
        return instance;
    }

    public int getFrameLength() {
        return frameLength;
    }

    public void setFrameLength(int frameLength) {
        this.frameLength = frameLength;
    }
}
