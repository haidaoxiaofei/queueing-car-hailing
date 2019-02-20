package riderdispatcher.simulator;

import riderdispatcher.estimator.Estimator;
import riderdispatcher.estimator.PoissonEstimator;

import java.io.IOException;

public class TestSimple {
    public static void main(String[] args) throws IOException {
        ClassLoader classLoader = TestSimple.class.getClassLoader();
        String confPath = classLoader.getResource("Simple.properties").getPath();
        EnvConf conf = new EnvConf(confPath);
        System.out.println(conf.gridHeightNum);
        System.out.println(conf.gridWidthNum);

        String driver_path = classLoader.getResource("simple_driver_arrival").getPath();
        String order_path = classLoader.getResource("simple_order_arrival").getPath();

        Estimator estimator = new PoissonEstimator(conf, driver_path, order_path);
//        Estimator estimator = new Estimator(conf, driver_path, order_path);

//        Env env = new Env(conf);

//        Status status = env.reset();
//        for(GridRequest gridReq : status.getGridList()){
//
//        }
    }
}
