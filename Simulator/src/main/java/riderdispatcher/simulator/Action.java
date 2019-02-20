package riderdispatcher.simulator;

import java.util.List;

public class Action {
    private List<GridResponse> response;

    public Action(List<GridResponse> response) {
        this.response = response;
    }

    public List<GridResponse> getResponse() {
        return response;
    }

    public void setResponse(List<GridResponse> response) {
        this.response = response;
    }
}
