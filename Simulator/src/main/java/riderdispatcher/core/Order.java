package riderdispatcher.core;

public class Order implements SavableObject, ZoneObj {
    private int id;
    private int startZoneID;
    private int endZoneID;
    private Point startPoint;
    private Point endPoint;
    private long startTime;
    private long endTime;
    private double cost;
    private double tripLength;
    private double maxWaitTime;
    private Driver driver = null;

    public static Point fakePoint = new Point(0,0,0);

    public Order(){
//        startPoint = fakePoint;
//        endPoint = fakePoint;
    }


    public Order(OrderRecord record, int id){
        this.id = id;
        this.startZoneID = record.pickZoneID;
        this.endZoneID = record.dropZoneID;
        this.startPoint = fakePoint;
        this.endPoint = fakePoint;
        this.startTime = record.pickTime;
        this.endTime = record.dropTime;
        this.cost = record.totalAmount;
        this.tripLength = record.tripDistance;
    }

    public Order(int id, int startGridid, int endGridid, Point startPoint, Point endPoint, int startTime, int endTime, int cost, double tripLength, int maxWaitTime) {
        this.id = id;
        this.startZoneID = startGridid;
        this.endZoneID = endGridid;
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.startTime = startTime;
        this.endTime = endTime;
        this.cost = cost;
        this.tripLength = tripLength;
        this.maxWaitTime = maxWaitTime;
    }

    public Order(Order order) {
        this.id = order.id;
        this.startZoneID = order.startZoneID;
        this.endZoneID = order.endZoneID;
        this.startPoint = order.startPoint;
        this.endPoint = order.endPoint;
        this.startTime = order.startTime;
        this.endTime = order.endTime;
        this.cost = order.cost;
        this.tripLength = order.tripLength;
        this.maxWaitTime = order.maxWaitTime;
    }

    public void assignDriver(Driver driver){
        this.driver = driver;
    }

    public void withdrawDriver(Driver driver){
        if (this.driver != null && this.driver != driver){
            System.out.println("Bad withdraw driver");
        } else {
            this.driver = null;
        }
    }

    public boolean isAssigned(){
        if (this.driver != null){
            return true;
        } else {
            return false;
        }
    }

    public boolean isExpired(long currentTimeOffset){
        if (this.startTime + this.maxWaitTime > currentTimeOffset){
            return false;
        } else {
            return true;
        }
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public int getStartZoneID() {
        return startZoneID;
    }

    public void setStartZoneID(int startGridid) {
        this.startZoneID = startGridid;
    }

    public int getEndZoneID() {
        return endZoneID;
    }

    public void setEndZoneID(int endZoneID) {
        this.endZoneID = endZoneID;
    }

    public Point getStartPoint() {
        return startPoint;
    }

    public void setStartPoint(Point startPoint) {
        this.startPoint = startPoint;
    }

    public Point getEndPoint() {
        return endPoint;
    }

    public void setEndPoint(Point endPoint) {
        this.endPoint = endPoint;
    }

    public double getTripLength() {
        return tripLength;
    }

    public void setTripLength(double tripLength) {
        this.tripLength = tripLength;
    }

    public double getMaxWaitTime() {
        return maxWaitTime;
    }

    public void setMaxWaitTime(double maxWaitTime) {
        this.maxWaitTime = maxWaitTime;
    }


    public long getStartTime() {
        return startTime;
    }

    public void setStartTime(long startTime) {
        this.startTime = startTime;
    }

    public long getEndTime() {
        return endTime;
    }

    public void setEndTime(long endTime) {
        this.endTime = endTime;
    }

    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }


    @Override
    public Order fromString(String recordString){
        String[] result = recordString.split(",");
        Order order = new Order();

        if (result.length <= 1) return null;
        order.id = Integer.valueOf(result[0]);
        order.startZoneID = Integer.valueOf(result[1]);
        order.endZoneID = Integer.valueOf(result[2]);
        if (startPoint != null && endPoint != null){
            order.startPoint.lng = Double.valueOf(result[3]);
            order.startPoint.lat = Double.valueOf(result[4]);
            order.endPoint.lng = Double.valueOf(result[5]);
            order.endPoint.lat = Double.valueOf(result[6]);
        }


        order.startTime = Long.valueOf(result[7]);
        order.endTime = Long.valueOf(result[8]);
        order.cost = Double.valueOf(result[9]);
        order.tripLength = Double.valueOf(result[10]);
        order.maxWaitTime = Double.valueOf(result[11]);
        return order;
    }

    @Override
    public String convertToString() {
        StringBuffer orderString = new StringBuffer();
        orderString.append(this.id).append(",")
                .append(this.startZoneID).append(",")
                .append(this.endZoneID).append(",")
                .append(this.startPoint==null?0:this.startPoint.lng).append(",")
                .append(this.startPoint==null?0:this.startPoint.lat).append(",")
                .append(this.endPoint==null?0:this.endPoint.lng).append(",")
                .append(this.endPoint==null?0:this.endPoint.lat).append(",")
                .append(Long.toString(this.startTime)).append(",")
                .append(Long.toString(this.endTime)).append(",")
                .append(this.cost).append(",")
                .append(this.tripLength).append(",")
                .append(Double.toString(this.maxWaitTime));
        return orderString.toString();
    }

    @Override
    public int getCurrentZoneID() {
        return this.startZoneID;
    }

    @Override
    public void setCurrentZoneID(int currentZoneID) {
        this.startZoneID = currentZoneID;
    }
}
