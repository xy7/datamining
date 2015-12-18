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
import java.util.Map;
import java.util.TreeMap;

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
		
		featureVector.setQuick(0, 1.0);//填充常量 k0
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

		OnlineLogisticRegression lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(50);// 1e-3
		lr.alpha(1 - 1.0e-3);// 学习率的指数衰减率

		File dir = new File("D:/bigdata/data/lost");
		File[] files = dir.listFiles();
		
		int passes = 20;
		analysisFiles(files, new LineHandler(){
			@Override
			public boolean handle(String line) {
				// TODO Auto-generated method stub

				Vector input = new RandomAccessSparseVector(numFeatures);
				int targetValue = processLine(line, input);
				// check performance while this is still news
				double logP = lr.logLikelihood(targetValue, input);
			
				double p = lr.classifyScalar(input);
				if (scores) {
					output.printf(Locale.ENGLISH, "%2d  %1.2f  |  %2.4f %10.4f%n",
							targetValue, p, lr.currentLearningRate(), logP);
				}

				// now update model
				lr.train(targetValue, input);
				return true;
			}
		}, passes);
		
		output.printf(Locale.ENGLISH, "%s %n", lr.getBeta().toString());
		try (DataOutputStream modelOutput = new DataOutputStream(new FileOutputStream("D:/bigdata/data/lrParam.txt"))) {
			lr.write(modelOutput);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		Integer[] res = {0, 0, 0, 0};//int all = 0, lost = 0, preLost = 0, right = 0;
		
		analysisFiles(files, new LineHandler(){
			
			@Override
			public boolean handle(String line) {
				Vector input = new RandomAccessSparseVector(numFeatures);
				int targetValue = processLine(line, input);
				
				double score = lr.classifyScalar(input);
				int predictValue = score > 0.5 ? 1 : 0;
				res[0]++;
				if (targetValue == 1) {
					res[1]++;
					if (predictValue == 1) {
						res[2]++;
					}
				}

				if (predictValue == targetValue) {
					res[3]++;
				}
				return true;
			}
		});

		double coverRate = (double) res[2] / res[1];
		double rightRate = (double) res[3] / res[0];
		output.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f %n"
				, coverRate, rightRate);

	}
	
	private static void analysisFiles(File[] files, LineHandler handler){
		analysisFiles(files, handler, 1);
	}
	
	private static void analysisFiles(File[] files, LineHandler handler, int passes){
		
		for (int pass = 0; pass < passes; pass++) {
			int all = 0;
			int suc = 0;
			for (File file : files) {
				LineIterator it = null;
				try {
					it = FileUtils.lineIterator(file);
					while (it.hasNext()) {
						String line = it.nextLine();
						all++;
						if(handler.handle(line) )
							suc++;
					}// while lines
				} catch (IOException e) {
					output.printf("!!!file read failed: %s %n", e);
				} finally {
					if (it != null)
						LineIterator.closeQuietly(it);
				}
			}// for file
			
			output.printf(Locale.ENGLISH, "pass %d: all(%d) sucess(%d) %n"
					, pass, all, suc);
		}// for pass
	}

}
