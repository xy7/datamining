package com.seasun.data.datamining;

import java.util.List;
import java.util.Random;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

public class KmeansUtil {

	public static void main(String[] args) {
		
		for(int i=0;i<100;i++){
			int[] res = random(10, 4);
			for(int j=0;j<4;j++)
				System.out.print(" j: " + res[j]);
			System.out.println("----");
		}

	}// main
	
	//循环找出最佳的k
//	public static List<Cluster> kmeansClass(List<Vector> inputs, double convergenceDelta) {
//		int k = 2;
//		List<Cluster> clustersPre = kmeansClass(inputs, k, convergenceDelta);
//		double avgRadiusPre = computeAvgRadius(clustersPre);
//		while(k<10){
//			k++;
//			List<Cluster> clustersThis = kmeansClass(inputs, k, convergenceDelta);
//			double avgRadiusThis = computeAvgRadius(clustersThis);
//			double delta = avgRadiusThis - avgRadiusPre;
//		}
//		return null;
//	}
	
	public static double computeAvgRadius(List<Cluster> clusters){
		double sum = 0.0;
		int num = 0;
		for(Cluster c:clusters){
			sum += c.getRadius().norm(2)*c.getNumObservations();
			num += c.getNumObservations();
		}
		return sum/num;
	}

	public static ClusterClassifier kmeansClass(List<Vector> inputs, int k, double convergenceDelta) {
		//System.out.println(inputs);
		// double convergenceDelta = 0.001;
		List<Cluster> clusters = Lists.newArrayList();
		// KMeansUtil.configureWithClusterInfo(conf, clustersIn, clusters);
		int nextClusterId = 0;
		DistanceMeasure measure = ClassUtils.instantiateAs(SquaredEuclideanDistanceMeasure.class.getName(),
				DistanceMeasure.class);
		int[] randomIndex = random(inputs.size(), k);
		for (int i:randomIndex) {
			Kluster newCluster = new Kluster(inputs.get(i), nextClusterId++, measure);
			clusters.add(newCluster);
		}

		ClusteringPolicy policy = new KMeansClusteringPolicy(convergenceDelta);
		ClusterClassifier classifier = new ClusterClassifier(clusters, policy);

		int iteration = 1;
		int numIterations = 100;

		while (iteration <= numIterations) {
			for (Vector vector : inputs) {
				// classification yields probabilities
				Vector probabilities = classifier.classify(vector);
				// policy selects weights for models given those probabilities
				Vector weights = classifier.getPolicy().select(probabilities);
				// training causes all models to observe data
				for (Vector.Element e : weights.nonZeroes()) {
					int index = e.index();
					classifier.train(index, vector, weights.get(index));//计算s0， s1， s2
				}
			}
			// compute the posterior models 计算是否收敛，计算质心和半径，重置s0， s1， s2
			classifier.close();
			// update the policy, here do nothing
			classifier.getPolicy().update(classifier);

			if(iteration%20 == 0)
				System.out.println("iteration:" + iteration + " Cluster:" + classifier.getModels());
			iteration++;

			if( isConverged(classifier) )
				break;

		}// while (iteration <= numIterations)

		return classifier;
	}

	public static boolean isConverged(ClusterClassifier classifier) {
		for (Cluster c : classifier.getModels()) {
			if (!c.isConverged())
				return false;
		}
		return true;
	}

	//产生K个递增的随机数
	public static int[] random(int size, int k){
		 Random random = RandomUtils.getRandom();
		 int[] res = new int[k];

		 int min = 0;
		 for(int i=0;i<k;i++){
			 int value = random.nextInt(size - (k-i) - min + 1);
			 res[i] = value + min;
			 min = res[i] + 1;
		 }
		 return res;
	}
}
