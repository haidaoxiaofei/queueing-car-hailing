package riderdispatcher.core;

import java.util.ArrayList;
import java.util.List;

public class ZoneIndex<ZoneObject> {
    List<List<ZoneObject>> index = new ArrayList<>();

    public ZoneIndex(List<ZoneObject> orders){
        for (int i = 0; i < ZoneDemandTable.TOTAL_ZONE_COUNT; i++) {
            index.add(new ArrayList());
        }

        for (ZoneObject obj: orders){
            index.get(((ZoneObj)obj).getCurrentZoneID() - 1).add(obj);
        }
    }

    public List<ZoneObject> queryZoneObjects(int zoneID){
        return index.get(zoneID - 1);
    }
}
