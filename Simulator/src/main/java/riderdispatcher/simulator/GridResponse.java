package riderdispatcher.simulator;

import java.util.List;

public class GridResponse {
    public static class ODPair{
        public int driverid;
        public int orderid;
    }

    private int gridid;
    private List<ODPair> odPairList;

    public int getGridid() {
        return gridid;
    }

    public void setGridid(int gridid) {
        this.gridid = gridid;
    }

    public GridResponse(List<ODPair> odPairList) {
        this.odPairList = odPairList;
    }

    public List<ODPair> getOdPairList() {
        return odPairList;
    }

    public void setOdPairList(List<ODPair> odPairList) {
        this.odPairList = odPairList;
    }
}
