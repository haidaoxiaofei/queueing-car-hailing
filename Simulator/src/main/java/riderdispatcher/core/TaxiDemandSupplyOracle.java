package riderdispatcher.core;

import java.text.ParseException;
import java.util.List;

public class TaxiDemandSupplyOracle {
//    private long baseTimeMS = TimeUtil.format.parse("2018-06-01 00:00:00").getTime()/1000;
    private long frameTime;
    private List<ZoneDemandTable> zoneDemandTables;


    public TaxiDemandSupplyOracle(List<ZoneDemandTable> zoneDemandTables, long frameTime) throws ParseException {
        this.zoneDemandTables = zoneDemandTables;
        this.frameTime = frameTime;
    }

    //if used for taxi plan records initial time frames here
    public void initialTimeFrames(){
        long totalTime = 24*60*60;
        long currentTimeFrameHead = 0;

        while (currentTimeFrameHead < totalTime){
            zoneDemandTables.add(new ZoneDemandTable());
            currentTimeFrameHead += frameTime;
        }
    }

    public double queryDemand(long startOffset, long endOffset, int zoneID){
        double totalDemand = 0;

        int startIndex = (int)Math.floor(startOffset / frameTime);
        int endIndex = (int)Math.floor(endOffset / frameTime);

        for (int i = startIndex; i <= endIndex; i++) {
            if (i >= zoneDemandTables.size()){
                break;
            }
            totalDemand += zoneDemandTables.get(i).getZoneDemand(zoneID);
        }

        double startAdjustRatio = (startOffset % frameTime)/(double)frameTime;
        double endAdjustRatio = 1 - (endOffset % frameTime)/(double)frameTime;

        if (startIndex < zoneDemandTables.size()){
            totalDemand -= - startAdjustRatio * zoneDemandTables.get(startIndex).getZoneDemand(zoneID);
        }
        if (endIndex < zoneDemandTables.size()){
            totalDemand -= endAdjustRatio * zoneDemandTables.get(endIndex).getZoneDemand(zoneID);
        }
        return totalDemand;
    }

    public double queryRate(long startOffset, long endOffset, int zoneID){
        double demand = queryDemand(startOffset, endOffset, zoneID);
        if (endOffset <= startOffset){
            return 0;
        } else {
            return demand / (endOffset - startOffset);
        }
    }




    public void addTimeRecord(int zoneID, long timeOffset, double demand){
        int index = (int)(timeOffset/frameTime);
        if (index >= zoneDemandTables.size()){
            return;
        }
        ZoneDemandTable table = zoneDemandTables.get(index);
        table.addDemand(zoneID, demand);
    }

    public void removeTimeRecord(int zoneID, long timeOffset, double demand){
        int index = (int)(timeOffset/frameTime);
        if (index >= zoneDemandTables.size()){
            return;
        }
        ZoneDemandTable table = zoneDemandTables.get(index);
        table.addDemand(zoneID, -demand);
    }
}
