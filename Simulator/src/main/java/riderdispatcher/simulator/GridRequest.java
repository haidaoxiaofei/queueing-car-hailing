package riderdispatcher.simulator;

import riderdispatcher.core.Driver;
import riderdispatcher.core.Order;

import java.util.List;

public class GridRequest {
    private int gridid;
    private List<Driver> driverList;
    private List<Order> orderList;

    public GridRequest(List<Driver> driverList, List<Order> orderList) {
        this.driverList = driverList;
        this.orderList = orderList;
    }

    public int getGridid() {
        return gridid;
    }

    public void setGridid(int gridid) {
        this.gridid = gridid;
    }

    public List<Driver> getDriverList() {
        return driverList;
    }

    public void setDriverList(List<Driver> driverList) {
        this.driverList = driverList;
    }
}
