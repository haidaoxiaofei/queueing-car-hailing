package riderdispatcher.core;

import java.util.ArrayList;
import java.util.List;

public class ZoneDemandTable implements SavableObject{
    public static final int TOTAL_ZONE_COUNT = 263;

    private List<Double> zoneDemandTable = new ArrayList<>();

    public ZoneDemandTable(){
        initZoneDemandTable();
    }

    public long getTotalDemand(){
        return (long)zoneDemandTable.stream().mapToDouble(Double::doubleValue).sum();
    }

    public List<Double> getDemandDistribution(){
        List<Double> demandDistribution = new ArrayList<>();

        long totalDemand = getTotalDemand();

        for (Double demand: zoneDemandTable){
            demandDistribution.add(demand/totalDemand);
        }

        return demandDistribution;
    }

    public void initZoneDemandTable(){
        for (int i = 0; i < TOTAL_ZONE_COUNT; i++) {
            zoneDemandTable.add(0d);
        }
    }

    public void addDemand(int zoneID, double demand){
        zoneDemandTable.set(zoneID - 1, zoneDemandTable.get(zoneID - 1) + demand);
    }

    public double getZoneDemand(int zoneID){
        return zoneDemandTable.get(zoneID - 1);
    }

    @Override
    public ZoneDemandTable fromString(String objectString) {
        String[] result = objectString.split(",");
        ZoneDemandTable demandTable = new ZoneDemandTable();

        if (result.length <= 1) return null;

        int offset = TOTAL_ZONE_COUNT - result.length;
        for (int i = offset; i < TOTAL_ZONE_COUNT; i++){
            demandTable.addDemand(i+1, Double.valueOf(result[i - offset]));
        }

        return demandTable;
    }

    @Override
    public String convertToString() {
        StringBuffer orderString = new StringBuffer();
        for (int i = 0; i < zoneDemandTable.size(); i++) {

            orderString.append(zoneDemandTable.get(i));

            if (i < zoneDemandTable.size() - 1){
                orderString.append(",");
            }
        }

        return orderString.toString();
    }
}
