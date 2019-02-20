/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package riderdispatcher.core;

import java.util.ArrayList;
import java.util.List;


/**
 *
 * @author bigstone
 */
public class ProblemInstance {
    public List<Driver> drivers = new ArrayList();
    public List<Order> orders = new ArrayList();

    public List<Order> completedOrders = new ArrayList<>();
    public List<Order> expiredOrders = new ArrayList<>();

    public String info;
    public String solverName;

    public TaxiDemandSupplyOracle orderReal;
    public TaxiDemandSupplyOracle orderOracle;
    public TaxiDemandSupplyOracle taxiWatchdog;

    public long currentTimeOffset;
    public long startRunningMillis;
    public long endRunningMillis;



    public ProblemInstance(ProblemInstance instance) {
        this.info = instance.info;
        this.startRunningMillis = instance.startRunningMillis;
        this.endRunningMillis = instance.endRunningMillis;
        this.solverName = instance.solverName;
        this.orderOracle = instance.orderOracle;
        this.orderReal = instance.orderReal;
        this.taxiWatchdog = instance.taxiWatchdog;

        for (int i = 0; i < instance.orders.size(); i++) {
            Order currentOrder = instance.orders.get(i);
            Order newOrder = new Order(currentOrder);
            this.orders.add(newOrder);
        }


        for (int i = 0; i < instance.drivers.size(); i++) {
            Driver currentDriver = instance.drivers.get(i);
            Driver newWorker = new Driver(currentDriver);
            this.drivers.add(newWorker);
        }

        for (int i = 0; i < instance.completedOrders.size(); i++) {

            Order newOrder = new Order(instance.completedOrders.get(i));
            this.completedOrders.add(newOrder);
        }

        for (int i = 0; i < instance.expiredOrders.size(); i++) {

            Order newOrder = new Order(instance.expiredOrders.get(i));
            this.expiredOrders.add(newOrder);
        }

    }
    
    
    
    public ProblemInstance() {

    }


    public double calculateTotalRevenue(){
        double totalRevenue = 0;

        for (Driver driver: drivers){
            for (Order order: driver.getOrders()){
                totalRevenue += order.getCost();
            }
        }
        return totalRevenue;
    }

    public double calculateTotalDistance(){
        double totalDistance = 0;

        for (Driver driver: drivers){
            for (Order order: driver.getOrders()){
                totalDistance += order.getTripLength();
            }
        }
        return totalDistance;
    }

    public double calculateTotalServingTime(){
        double totalTime = 0;

        for (Driver driver: drivers){
            for (Order order: driver.getOrders()){
                totalTime += order.getEndTime() - order.getStartTime();
            }
        }
        return totalTime;
    }

    public int calculateTotalAssignedOrderCount(){
        int totalAssignedOrderCount = 0;
        for (Driver driver: drivers){
            totalAssignedOrderCount += driver.getOrders().size();
        }

        return totalAssignedOrderCount;
    }

    
    
}
