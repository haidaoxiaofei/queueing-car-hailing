package riderdispatcher.simulator;

import java.util.List;

public class Status {
    private List<GridRequest> gridList;

    public Status(List<GridRequest> gridList) {
        this.gridList = gridList;
    }

    public List<GridRequest> getGridList() {
        return gridList;
    }

    public void setGridList(List<GridRequest> gridList) {
        this.gridList = gridList;
    }
}
