package com.seasun.data.datamining;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.io.Charsets;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;

public class Utils {
	
	private static Properties props = new Properties();
	private static boolean loadFile = false;
	
	public static double CLASSIFY_VALUE = 0.5;
	public static boolean scores = true;
	private static String MODEL_PARAM_FILE = "lrParam.txt";

	private static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);

	public static boolean loadConfigFile(String file){
		if(loadFile)
			return true;
		
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
	    
	    loadFile = true;
	    
	    CLASSIFY_VALUE = getOrDefault("classify_value", CLASSIFY_VALUE);
	    scores = getOrDefault("scores", scores);
	    MODEL_PARAM_FILE = getOrDefault("model_param_file", MODEL_PARAM_FILE);

	    return true;
	}

	
	public static <T> T getOrDefault(String key, T def){
//		if(!def.getClass().isPrimitive())
//			return null;

		if(!loadFile)
			loadConfigFile("./config.properties");
		
		String value = props.getProperty(key);
		if(value == null){
			out.printf("get %s default: %s %n", key, def);
			return def;
		}
		
		out.printf("get %s : %s %n", key, value);
		
		String typeName = def.getClass().getName();
		if(typeName.equals("java.lang.Double")){
			Double res = Double.parseDouble(value);
			return (T)res;
		} else if(typeName.equals("java.lang.Integer")){
			Integer res = Integer.parseInt(value);
			return (T)res;
		} else if(typeName.equals("java.lang.Long")){
			Long res = Long.parseLong(value);
			return (T)res;
		} else if(typeName.equals("java.lang.Float")){
			Float res = Float.parseFloat(value);
			return (T)res;
		} else if(typeName.equals("java.lang.String")){
			return (T)value;
		} else if(typeName.equals("java.lang.Boolean")){
			Boolean res = Boolean.parseBoolean(value);
			return (T)res;
		}
		
		return null;
	}
	
	public static boolean saveModel(OnlineLogisticRegression lr){
		
		out.printf(Locale.ENGLISH, "%s %n", lr.getBeta().toString());
		
		try (DataOutputStream modelOutput = new DataOutputStream(new FileOutputStream(MODEL_PARAM_FILE))) {
			lr.write(modelOutput);
			modelOutput.close();
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public static OnlineLogisticRegression loadModelParam() {
		OnlineLogisticRegression lr = new OnlineLogisticRegression();
		
		InputStream input = null;
		try {
			input = new FileInputStream(MODEL_PARAM_FILE);
			DataInput in = new DataInputStream(input);
			lr.readFields(in);
			input.close();
		} catch (Exception e) {
			e.printStackTrace();
			out.println("load model param file failed!");
		}
		return lr;
	}
	
	//将多天的统计结果打印出来
	public static void printEvalRes(Map<String, double[]> resMap) {
		double avgRes[] = {0.0, 0.0, 0.0};
		double minRes[] = {1.0, 1.0, 1.0};
		List<String> dates = new LinkedList<>(resMap.keySet());
		dates.sort(new Comparator<String>(){

			@Override
			public int compare(String o1, String o2) {
				// TODO Auto-generated method stub
				return o1.compareTo(o2);
			}
			
		});
		
		for(String name:dates){
			double[] res = resMap.get(name);
			out.printf(Locale.ENGLISH, "dir name:%s	cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
					, name , res[0], res[1], res[2]);
			
			for(int i=0;i<3;i++){
				avgRes[i] += res[i];
				if(res[i] < minRes[i])
					minRes[i] = res[i];
			}
		}//for map
		
		int n = resMap.size();
		for(int i=0;i<3;i++){
			avgRes[i] /= n;
		}
		
		out.printf(Locale.ENGLISH, "avg	cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
				, avgRes[0], avgRes[1], avgRes[2]);
		out.printf(Locale.ENGLISH, "min	cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
				, minRes[0], minRes[1], minRes[2]);
	}
	
	//return day1 - day2 + 1
	public static int dateDiff(LocalDate day1, LocalDate day2){
		if(day1.equals(day2)){
			return 0;
		} else if(day1.isAfter(day2) ){
			int i = 0;
			for(LocalDate ld=day2; !ld.isAfter(day1); ld=ld.plusDays(1)){
				i++;
			}
			return i;
		} else if(day1.isBefore(day2)) {
			int i = 0;
			for(LocalDate ld=day1; !ld.isAfter(day2); ld=ld.plusDays(1)){
				i++;
			}
			return -1*i;
		}
		return 0;
	}
	
	public static void main(String[] args) {
		int i = 10;
		out.printf("get default: %s %n",  1);
	}
}
