package riderdispatcher.estimator;

public class IdleTimeEstimator {

    public static double estimateIdleTime(double lambda, double mu, int K){
        if (lambda == 0){
            return 1;
        }
        if (mu == 0){
            mu = lambda/10000;
        }
        if (lambda>=mu){
            return Math.max(K/mu, lambda/(mu*0.000001));
        } else {
            return lambda/(mu*(mu-lambda));
        }
    }
}
