package riderdispatcher.core;

import java.util.ArrayList;
import java.util.List;

public class Driver implements SavableObject, ZoneObj{
    public int id;
    private Point curPos;
    private List<Order> orders = new ArrayList<>();
    private Order servingOrder = null;
    private long nextFreeTimeOffset;
    private int currentZoneID;

    public Driver(){

    }

    public Driver(Driver driver) {
        this.id = driver.id;
        this.curPos = driver.curPos;

        this.servingOrder = driver.servingOrder;
        this.nextFreeTimeOffset = driver.nextFreeTimeOffset;
        this.currentZoneID = driver.currentZoneID;

        this.orders = new ArrayList<>(driver.orders);
    }

    public Driver(int id, Point curPos, List<Order> orders, Order servingOrder, int nextFreeTime, int currentGridId) {
        this.id = id;
        this.curPos = curPos;
        this.orders = orders;
        this.servingOrder = servingOrder;
        this.nextFreeTimeOffset = nextFreeTime;
        this.currentZoneID = currentGridId;
    }

    public void serveOrder(Order order, long currentTimeOffset) {
//        if (this.isBusy(currentTimeOffset)){
//            System.out.println("Busy Assign!");
//        }

        if (order.isAssigned()){
            System.out.println("Double Assign!");
        }
        order.assignDriver(this);
        servingOrder = order;
        this.getOrders().add(order);
        this.nextFreeTimeOffset = currentTimeOffset + order.getEndTime() - order.getStartTime();
    }

    public void withdrawOrder(Order order){
        if (servingOrder!=null && servingOrder != order){
            System.out.println("Bad withdraw order!");
        } else {
            orders.remove(servingOrder);
            servingOrder.withdrawDriver(this);
            servingOrder = null;
            this.nextFreeTimeOffset = this.nextFreeTimeOffset - (order.getEndTime() - order.getStartTime());
        }
    }

    public boolean isBusy(long currentTimeOffset){
        return this.nextFreeTimeOffset > currentTimeOffset;
    }
    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public double getNextFreeTimeOffset() {
        return nextFreeTimeOffset;
    }

    public void setNextFreeTimeOffset(long nextFreeTimeOffset) {
        this.nextFreeTimeOffset = nextFreeTimeOffset;
    }

    public List<Order> getOrders() {
        return orders;
    }

    public void setOrders(List<Order> orders) {
        this.orders = orders;
    }

    public Order getServingOrder() {
        return servingOrder;
    }

    public void setServingOrder(Order servingOrder) {
        this.servingOrder = servingOrder;
    }


    public Point getCurPos() {
        return curPos;
    }

    public void setCurPos(Point curPos) {
        this.curPos = curPos;
    }

    @Override
    public Driver fromString(String recordString){
        String[] result = recordString.split(",");
        Driver driver = new Driver();

        if (result.length <= 1) return null;
        driver.id = Integer.valueOf(result[0]);
        if (driver.curPos != null){
            driver.curPos.lng = Double.valueOf(result[1]);
            driver.curPos.lat = Double.valueOf(result[2]);
            driver.curPos.time = Long.valueOf(result[3]);
        }

        driver.nextFreeTimeOffset = Long.valueOf(result[4]);
        driver.currentZoneID = Integer.valueOf(result[5]);
        driver.servingOrder = null;
        driver.orders = new ArrayList<>();

        return driver;
    }

    @Override
    public String convertToString() {
        StringBuffer orderString = new StringBuffer();
        orderString.append(this.id).append(",")
                .append(this.curPos==null?0:this.curPos.lng).append(",")
                .append(this.curPos==null?0:this.curPos.lat).append(",")
                .append(this.curPos==null?0:this.curPos.time).append(",")
                .append(Long.toString(this.nextFreeTimeOffset)).append(",")
                .append(this.currentZoneID);

        return orderString.toString();
    }

    @Override
    public int getCurrentZoneID() {
        return currentZoneID;
    }

    @Override
    public void setCurrentZoneID(int currentZoneID){
        this.currentZoneID = currentZoneID;
    }
}
