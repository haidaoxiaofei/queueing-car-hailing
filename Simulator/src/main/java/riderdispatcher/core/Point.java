package riderdispatcher.core;

public class Point {
    public double lng;
    public double lat;
    public long time;


    public Point(double lng, double lat) {
        this.lng = lng;
        this.lat = lat;
    }

    public Point(double lng, double lat, long time) {
        this.lng = lng;
        this.lat = lat;
        this.time = time;
    }


}
