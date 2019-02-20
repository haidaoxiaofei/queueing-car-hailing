package riderdispatcher.simulator;


import riderdispatcher.algorithms.AlgorithmEngine;
import riderdispatcher.estimator.IdleTimeEstimator;

public class Main {
    public static void main(String[] args) throws Exception {
        AlgorithmEngine engine = new AlgorithmEngine();
        engine.run();

//        for (int i = 1; i < 10; i++) {
//            double mu = i;
//            for (int j = 1; j < 5; j++) {
//
//                double lambda = j * mu;
//                int K = (int)mu;
//
//
//                double idleTime = IdleTimeEstimator.estimateIdleTime(lambda, mu, K);
//                double pAcc = IdleTimeEstimator.positiveAccumulateValue(lambda, mu);
//                double p0 = IdleTimeEstimator.pZore(lambda, mu, K);
//                double pLB = IdleTimeEstimator.pLeftBottom(lambda,mu,K);
//                double ratio = IdleTimeEstimator.ratio(lambda,mu,K);
//                double rn = IdleTimeEstimator.rn(j,mu);
//                System.out.print("idleTime:" + idleTime+ " pAcc:"+ pAcc + " p0:"+p0+" pLB:"+ pLB+" ratio:"+ratio+" rn:"+rn+"\n");
//
//            }
//            System.out.println();
//            System.out.println();
//        }


    }
}
