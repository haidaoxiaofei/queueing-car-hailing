package riderdispatcher.algorithms;

import riderdispatcher.core.Driver;
import riderdispatcher.core.Order;
import riderdispatcher.core.ProblemInstance;

public class UpperAlgorithm extends BaseAlgorithm {
    private String info = "UpperAlgorithm";

    @Override
    public void run(ProblemInstance instance) {
        Driver driver = instance.drivers.get(0);
        for (Order order: instance.orders){
            driver.serveOrder(order, instance.currentTimeOffset);
        }
    }

    @Override
    public String getInfo() {
        return info;
    }
}
