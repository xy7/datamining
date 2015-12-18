package com.seasun.data.datamining;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Hello world!
 *
 */
public class LogisticRegressionTrain
{
	private static final String COLUMN_SPLIT = "\t";
	
	private static boolean scores = true;
	private static PrintWriter output = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
	private static int numFeatures = 6;

	public static int processLine(String line, Vector featureVector) {
		featureVector.setQuick(0, 1.0);
		List<String> values = Arrays.asList(line.split(COLUMN_SPLIT));
		int res = 0;
		for (int i = 0; i < values.size(); i++) {
			if (i == values.size() - 1) {
				res = Integer.parseInt(values.get(i));
				continue;
			}

			featureVector.setQuick(i + 1, Double.parseDouble(values.get(i)));
		}

		if (res != 1)
			res = 0;
		return res;
	}

	public static void main(String[] args) {
		int passes = 20;

		OnlineLogisticRegression lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(50);// 1e-3
		lr.alpha(1 - 1.0e-3);// 学习率的指数衰减率

		File dir = new File("D:/bigdata/data/lost");
		File[] files = dir.listFiles();

		trainModelForFiles(passes, lr, files);
		estimateEffect(lr, files);
	}

	private static void trainModelForFiles(int passes, OnlineLogisticRegression lr, File[] files) {
		for (int pass = 0; pass < passes; pass++) {

			int samples = 0;

			for (File file : files) {
				LineIterator it = null;
				try {
					it = FileUtils.lineIterator(file);
					while (it.hasNext()) {
						String line = it.nextLine();
						trainModel(lr, samples, line);
					}// while lines
				} catch (IOException e) {
					System.out.println("!!!file read failed:" + e);
				} finally {
					if (it != null)
						LineIterator.closeQuietly(it);
				}
			}// for file
		}

		System.out.println(lr.getBeta());

		try (DataOutputStream modelOutput = new DataOutputStream(new FileOutputStream("D:/bigdata/lr_model_param.txt"))) {
			lr.write(modelOutput);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void trainModel(OnlineLogisticRegression lr, int samples, String line) {

		double logPEstimate = 0;
		Vector input = new RandomAccessSparseVector(numFeatures);
		int targetValue = processLine(line, input);
		// check performance while this is still news
		double logP = lr.logLikelihood(targetValue, input);
		if (!Double.isInfinite(logP)) {
			if (samples < 20) {
				logPEstimate = (samples * logPEstimate + logP) / (samples + 1);
			} else {
				logPEstimate = 0.95 * logPEstimate + 0.05 * logP;
			}
			samples++;
		}
		double p = lr.classifyScalar(input);
		if (scores) {
			output.printf(Locale.ENGLISH, "%10d: %2d  %1.2f  |  %2.4f %10.4f %10.4f%n",
					samples, targetValue, p, lr.currentLearningRate(), logP, logPEstimate);
		}

		// now update model
		lr.train(targetValue, input);
	}

	// 计算准确率和覆盖率
	private static void estimateEffect(OnlineLogisticRegression lr, File[] files) {

		int all = 0, lost = 0, preLost = 0, right = 0;
		for (File file : files) {
			LineIterator it = null;
			try {
				it = FileUtils.lineIterator(file);
				while (it.hasNext()) {
					String line = it.nextLine();
					String[] columns = line.split(COLUMN_SPLIT);
					Vector input = new RandomAccessSparseVector(numFeatures);
					int targetValue = processLine(line, input);
					
					double score = lr.classifyScalar(input);
					int predictValue = score > 0.5 ? 1 : 0;
					all++;
					if (targetValue == 1) {
						lost++;
						if (predictValue == 1) {
							preLost++;
						}
					}

					if (predictValue == targetValue) {
						right++;
					}
				}// while
			} catch (IOException e) {
				System.out.println("!!!file read failed:" + e);
			} finally {
				if (it != null)
					LineIterator.closeQuietly(it);
			}
		}// for files

		double coverRate = (double) preLost / lost;
		double rightRate = (double) right / all;

		output.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f %n",
				 coverRate, rightRate);
	}
}
