package com.seasun.data.datamining;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Hello world!
 *
 */
public class LogisticRegressionTrain{
	
	private static double CLASSIFY_VALUE = 0.5;
	private static String SAMPLE_DIR = "./fig_app_user/sample/";
	private static int TARIN_PASSES = 5;
	private static String PREDICT_DIR = "./fig_app_user/predict/";
	private static String MODEL_PARAM_FILE = "lrParam.txt";
	private static String COLUMN_SPLIT = "\1";
	private static boolean scores = true;
	private static PrintWriter output = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);

	//role_level first_login_cnt first_online_dur total_recharge total_recharge_cnt last7_login_cnt last7_login_daycnt last7_online_dur
	private static int[] index;
	private static int numFeatures;
	
	private static int parseLine(String line, Vector featureVector) throws Exception {
		featureVector.setQuick(0, 1.0);//填充常量 k0
		List<String> values = Arrays.asList(line.split(COLUMN_SPLIT));
		if(values.size() < index[index.length-1] + 1)
			throw new Exception("parse error, columns size to small: " + values.size());
		
		for (int i = 0; i < index.length; i++) {
			String s = values.get(index[i]);
			if(s.equals("\\N"))//null值替换为空值
				s = "0";
			featureVector.setQuick(i + 1, Double.parseDouble(s));
		}
		
		int res = Integer.parseInt(values.get(values.size()-1));
		if(res !=0 && res != 1){
			System.out.println("parse error: " + line);
			throw new Exception("parse error, res not in (0,1)");
		}
		return res;
	}
	
	private static boolean loadConfigFile(String file){
		Utils.loadConfigFile(file);
  
	    String indexStr = Utils.getOrDefault("index", "10,25,26,28,29,30,31,32,33,34,35,65");
	    if(indexStr != null){
	    	String[] indexArray = indexStr.split(",");
	    	index = new int[indexArray.length];
	    	for(int i=0;i<indexArray.length;i++){
	    		index[i] = Integer.parseInt(indexArray[i].trim());
	    	}
	    	numFeatures = index.length + 1;
	    	System.out.print("index:");
		    for(int i:index)
		    	System.out.print(", " + i);
		    System.out.println("");
	    }
	    
	    COLUMN_SPLIT = Utils.getOrDefault("column_split", COLUMN_SPLIT);
	    MODEL_PARAM_FILE = Utils.getOrDefault("model_param_file", MODEL_PARAM_FILE);
	    SAMPLE_DIR = Utils.getOrDefault("sample_dir", SAMPLE_DIR);
	    TARIN_PASSES = Utils.getOrDefault("train_passes", TARIN_PASSES);
	    PREDICT_DIR = Utils.getOrDefault("predict_dir", PREDICT_DIR);
	    scores = Utils.getOrDefault("scores", scores);
	    CLASSIFY_VALUE = Utils.getOrDefault("classify_value", CLASSIFY_VALUE);
	    return true;
	}

	private static void evalModel() {
		OnlineLogisticRegression lr = new OnlineLogisticRegression();
		Utils.loadModelParam(lr); 
		
		evalModel(lr, PREDICT_DIR);
	}
	
	private static void evalModelMutiDir(String dirStr){
	
		OnlineLogisticRegression lr = new OnlineLogisticRegression();
		Utils.loadModelParam(lr);
		
		File dir = new File(dirStr);
		File[] files = dir.listFiles();
		for(File file:files){
			if(file.isDirectory()){
				System.out.println("dir: " + file.getName());
				evalModel(lr, file.getAbsolutePath());
			}
		}
		
		Utils.printEvalRes(resMap);
	}
	
	private static Map<String, double[]> resMap = new HashMap<>();
	
	private static void evalModel(OnlineLogisticRegression lr, String inputDir){
		
		Auc collector = new Auc();
		Integer[] res = {0, 0, 0, 0};//abcd;
		
		File dir = new File(inputDir);
		File[] files = dir.listFiles();
		Utils.analysisFiles(files, new LineHandler(){

			@Override
			public boolean handle(String line) {
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
				collector.add(targetValue, score);
				int predictValue = score > CLASSIFY_VALUE ? 1 : 0;

				if (targetValue == 1) {
					if (predictValue == 1) {
						res[0]++; //流失用户预测正确
					} else {
						res[2]++; //流失用户预测错误
					}
				} else {
					if (predictValue == 1) {
						res[1]++; //非流失用户预测错误
					} else {
						res[3]++; //非流失用户预测正确
					}
				}
				return true;
			}
		});
		
		int all = res[0] + res[1] + res[2] + res[3];
		output.printf("result matrix: lostcnt:%d	remaincnt:%d%n", res[0]+res[2], res[1]+res[3]);
		output.printf("A:%2.4f	B:%2.4f %n", (double)res[0]/all, (double)res[1]/all);
		output.printf("C:%2.4f	D:%2.4f %n", (double)res[2]/all, (double)res[3]/all);

		double coverRate = (double) res[0]/(res[0]+res[2]);//覆盖率
		double rightRate = (double) (res[0]+res[3])/all;//正确率
		double hitRate = (double) res[0]/(res[0]+res[1]);//命中率
		output.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
				, coverRate, rightRate, hitRate);
		
		output.printf(Locale.ENGLISH, "AUC = %.2f%n", collector.auc());
		
		double[] tmp = {coverRate, rightRate, hitRate};
		resMap.put(dir.getName(), tmp);
	}
	
	public static void trainModelMutiDir(){
		File dirPar = new File(SAMPLE_DIR);
		File[] dirs = dirPar.listFiles();
		List<File> files = new LinkedList<>();
		for(File dir:dirs){
			if(dir.isDirectory()){
				System.out.println("dir: " + dir.getName());
				files.addAll( Arrays.asList( dir.listFiles() ) );
			}
		}
		
		OnlineLogisticRegression lr = trainFiles(files.toArray(new File[0]));
		evalModelMutiDir(SAMPLE_DIR);
	}
	
	private static void trainModel(){
		File dir = new File(SAMPLE_DIR);
		File[] files = dir.listFiles();
		
		OnlineLogisticRegression lr = trainFiles(files);
		
		evalModel(lr, SAMPLE_DIR);
	}

	private static OnlineLogisticRegression trainFiles(File[] files) {
		OnlineLogisticRegression lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(1e-1);// 1e-3
		lr.alpha(1 - 1.0e-5);// 学习率的指数衰减率,步长

		Utils.analysisFiles(files, new LineHandler(){

			@Override
			public boolean handle(String line) {
			
				Vector input = new RandomAccessSparseVector(numFeatures);
				int targetValue;
				try {
					targetValue = parseLine(line, input);
				} catch (Exception e) {
					//e.printStackTrace();
					return false;
				}
				
				if (scores) {
					// check performance while this is still news
					double logP = lr.logLikelihood(targetValue, input);
					double p = lr.classifyScalar(input);
					output.printf(Locale.ENGLISH, "%2d  %1.4f  |  %2.6f %10.4f%n",
							targetValue, p, lr.currentLearningRate(), logP);
				}

				// now update model
				lr.train(targetValue, input);
				return true;
			}
		}, TARIN_PASSES);

		Utils.saveModel(lr);

		return lr;
	}

	public static void main(String[] args) {
		
		if(!loadConfigFile("./config.properties")){
			if(!loadConfigFile("./config/config.properties")){
				System.out.println("load config file error");
				return;
			}
		}
		
		if(args.length >= 1){
			if(args[0].equals("train"))
				trainModel();
			else if(args[0].equals("predict"))
				evalModelMutiDir(PREDICT_DIR);
			else if(args[0].equals("muti")){
				trainModelMutiDir();
				evalModelMutiDir(PREDICT_DIR);
			}
		} else {
			trainModel();
			evalModel();
		}

	}
}
