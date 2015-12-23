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
	private static boolean scores = true;
	private static PrintWriter output = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);

	//role_level first_login_cnt first_online_dur total_recharge total_recharge_cnt last7_login_cnt last7_login_daycnt last7_online_dur
	private static int[] index = {10, 15, 16, 17, 25, 34, 43};
	private static int numFeatures = index.length + 1;

//	public static int parseLine(String line, Vector featureVector) {
//		
//		featureVector.setQuick(0, 1.0);//填充常量 k0
//		List<String> values = Arrays.asList(line.split("\t"));
//		int res = 0;
//		for (int i = 0; i < values.size(); i++) {
//			if (i == values.size() - 1) {
//				res = Integer.parseInt(values.get(i));
//				continue;
//			}
//
//			featureVector.setQuick(i + 1, Double.parseDouble(values.get(i)));
//		}
//
//		if (res != 1)
//			res = 0;
//		return res;
//	}
	
	public static int parseLine(String line, Vector featureVector) throws Exception {
		featureVector.setQuick(0, 1.0);//填充常量 k0
		List<String> values = Arrays.asList(line.split("\1"));
		if(values.size() < index[index.length-1] + 1)
			throw new Exception("parse error, columns size to small: " + values.size());
		
		for (int i = 0; i < index.length; i++) {
			String s = values.get(index[i]);
			if(s.equals("\\N"))
				s = "0";
			featureVector.setQuick(i + 1, Double.parseDouble(s));
		}
		
		int res = Integer.parseInt(values.get(65));
		if(res !=0 && res != 1){
			System.out.println("parse error: " + line);
			throw new Exception("parse error, res not in (0,1)");
		}
		return res;
	}

	public static void main(String[] args) {

		OnlineLogisticRegression lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(50);// 1e-3
		lr.alpha(1 - 1.0e-3);// 学习率的指数衰减率

		File dir = new File("D:/bigdata/data/xyfm/fig_app_user_2015-12-15/");
		File[] files = dir.listFiles();
		
		//一半样本用来训练，一半用来预测
		int half = 1313/2;
		int passes = 5;
		analysisFiles(files, new LineHandler(){
			int i = 0;
			@Override
			public boolean handle(String line) {
				i++;
				if(i>half)
					return false;
				Vector input = new RandomAccessSparseVector(numFeatures);
				int targetValue;
				try {
					targetValue = parseLine(line, input);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					return false;
				}
				
				if (scores) {
					// check performance while this is still news
					double logP = lr.logLikelihood(targetValue, input);
					double p = lr.classifyScalar(input);
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
			int i = 0;
			@Override
			public boolean handle(String line) {
				i++;
				if(i<=half)
					return false;
				Vector input = new RandomAccessSparseVector(numFeatures);
				int targetValue;
				try {
					targetValue = parseLine(line, input);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					return false;
				}
				
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
		output.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f  1cnt:%d  0cnt:%d %n"
				, coverRate, rightRate, res[1], res[0]-res[1]);

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
					it = FileUtils.lineIterator(file, "UTF-8");
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
