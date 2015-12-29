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
import java.util.List;
import java.util.Locale;
import java.util.Properties;

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
public class LogisticRegressionTrain{
	
	private static boolean loadFile = false;
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
	
	public static boolean loadConfigFile(String file){
		if(loadFile)
			return true;
		Properties props = new Properties();
	    //InputStream in = LogisticRegressionTrain.class.getResourceAsStream(file); //配置文件的相对路径以类文件所在目录作为当前目录
		InputStream in = null;
		try {
			in = new FileInputStream(file); //配置文件的相对路径以工作目录
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.out.println("read config file error");
			return false;
		}
	    
	    try {
			props.load(in);
			in.close();
		} catch (IOException e1) {
			e1.printStackTrace();
			System.out.println("load config file error");
			return false;
		}
  
	    String indexStr = props.getProperty("index");
	    if(indexStr != null){
	    	String[] indexArray = indexStr.split(",");
	    	index = new int[indexArray.length];
	    	for(int i=0;i<indexArray.length;i++){
	    		index[i] = Integer.parseInt(indexArray[i].trim());
	    	}
	    	numFeatures = index.length + 1;
	    	System.out.println("index:");
		    for(int i:index)
		    	System.out.println(i);
	    }
	    
	    COLUMN_SPLIT = replaceStringProp(props, "column_split", COLUMN_SPLIT);
	    MODEL_PARAM_FILE = replaceStringProp(props, "model_param_file", MODEL_PARAM_FILE);
	    SAMPLE_DIR = replaceStringProp(props, "sample_dir", SAMPLE_DIR);
	    TARIN_PASSES = Integer.parseInt( replaceStringProp(props, "train_passes", Integer.toString(TARIN_PASSES) ) );
	    PREDICT_DIR = replaceStringProp(props, "predict_dir", PREDICT_DIR);
	    scores = Boolean.parseBoolean(replaceStringProp(props, "scores", Boolean.toString(scores)) );
  
	    loadFile = true;
	    return true;
	}
	
	private static String replaceStringProp(Properties props, String key, String target){
		 String value = props.getProperty(key);
		    if(value != null)
		    	target = value;
		    System.out.println(key + ": " + target);
		   return target;
	}
	
	public static void evalModel() {
		OnlineLogisticRegression lr = new OnlineLogisticRegression();
		
		InputStream input = null;
		try {
			input = new FileInputStream(MODEL_PARAM_FILE);
			DataInput in = new DataInputStream(input);
			lr.readFields(in);
			input.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("load model param file failed!");
		} 
		
		evalModel(lr, PREDICT_DIR);
	}
	
	public static void evalModel(OnlineLogisticRegression lr, String inputDir){
		
		if(!loadConfigFile("./config.properties")){
			if(!loadConfigFile("./config/config.properties")){
				System.out.println("load config file error");
				return;
			}
		}
		
		Integer[] res = {0, 0, 0, 0};//abcd;
		
		File dir = new File(inputDir);
		File[] files = dir.listFiles();
		analysisFiles(files, new LineHandler(){

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
				int predictValue = score > 0.5 ? 1 : 0;

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
	}
	
	public static void trainModel(){
		if(!loadConfigFile("./config.properties")){
			if(!loadConfigFile("./config/config.properties")){
				System.out.println("load config file error");
				return;
			}
		}
		
		OnlineLogisticRegression lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(50);// 1e-3
		lr.alpha(1 - 1.0e-3);// 学习率的指数衰减率

		File dir = new File(SAMPLE_DIR);
		File[] files = dir.listFiles();

		analysisFiles(files, new LineHandler(){

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
					output.printf(Locale.ENGLISH, "%2d  %1.2f  |  %2.4f %10.4f%n",
							targetValue, p, lr.currentLearningRate(), logP);
				}

				// now update model
				lr.train(targetValue, input);
				return true;
			}
		}, TARIN_PASSES);
		
		output.printf(Locale.ENGLISH, "%s %n", lr.getBeta().toString());
		try (DataOutputStream modelOutput = new DataOutputStream(new FileOutputStream(MODEL_PARAM_FILE))) {
			lr.write(modelOutput);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		evalModel(lr, SAMPLE_DIR);
	}

	private static void analysisFiles(File[] files, LineHandler handler){
		analysisFiles(files, handler, 1);
	}
	
	private static void analysisFiles(File[] files, LineHandler handler, int passes){
		
		for (int pass = 0; pass < passes; pass++) {
			int all = 0;
			int suc = 0;
			for (File file : files) {
				if(file.getName().endsWith("crc"))
					continue;
				if(file.isDirectory())
					continue;
				
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

	public static void main(String[] args) {
		if(args.length >= 1){
			if(args[0].equals("train"))
				trainModel();
			else 
				evalModel();
		} else {
			trainModel();
			evalModel();
		}

	}
}
