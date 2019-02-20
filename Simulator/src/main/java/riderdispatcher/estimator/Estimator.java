package riderdispatcher.estimator;


import riderdispatcher.core.Driver;
import riderdispatcher.core.Order;

import java.util.List;

public interface Estimator {

    public List<Driver> genRandomDrivers(int gridid, int systemTime);

    public List<Order> genRandomOrders(int gridid, int systemTime);

    public boolean isDriverQuit(Driver d, int systemTime);

}
