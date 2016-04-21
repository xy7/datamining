package com.seasun.data.datamining;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.io.compress.CompressionInputStream;
import org.apache.mahout.classifier.sgd.AbstractOnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;

import com.github.dataswitch.util.HadoopConfUtil;

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
		if(def instanceof Double){
			Double res = Double.parseDouble(value);
			return (T)res;
		} else if(def instanceof Integer){
			Integer res = Integer.parseInt(value);
			return (T)res;
		} else if(def instanceof Long){
			Long res = Long.parseLong(value);
			return (T)res;
		} else if(def instanceof Float){
			Float res = Float.parseFloat(value);
			return (T)res;
		} else if(def instanceof String){
			return (T)value;
		} else if(def instanceof Boolean){
			Boolean res = Boolean.parseBoolean(value);
			return (T)res;
		}
		
		return null;
	}
	
	public static boolean saveModel(Writable lr){
		
		if(lr instanceof AbstractOnlineLogisticRegression)
			out.printf(Locale.ENGLISH, "%s %n", ((AbstractOnlineLogisticRegression)lr).getBeta().toString());
		
		try (DataOutputStream modelOutput = new DataOutputStream(new FileOutputStream(MODEL_PARAM_FILE))) {
			lr.write(modelOutput);
			modelOutput.close();
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public static <T> boolean loadModelParam(T lr) {

		InputStream input = null;
		
		try {
			input = new FileInputStream(MODEL_PARAM_FILE);
			DataInput in = new DataInputStream(input);
			if(lr instanceof CrossFoldLearner){
				((CrossFoldLearner)lr).readFields(in);
			} else if(lr instanceof OnlineLogisticRegression){
				((OnlineLogisticRegression)lr).readFields(in);
			} else {
				out.println("load model failed, can only load CrossFoldLearner or OnlineLogisticRegression");
				input.close();
				return false;
			}
			
			input.close();
		} catch (Exception e) {
			e.printStackTrace();
			out.println("load model param file failed!");
		}
		return true;
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
	
	//return day1 - day2
	public static int dateDiff(LocalDate day1, LocalDate day2){
		if(day1.equals(day2)){
			return 0;
		} else if(day1.isAfter(day2) ){
			int i = 0;
			for(LocalDate ld=day2; ld.isBefore(day1); ld=ld.plusDays(1)){
				i++;
			}
			return i;
		} else if(day1.isBefore(day2)) {
			int i = 0;
			for(LocalDate ld=day1; ld.isBefore(day2); ld=ld.plusDays(1)){
				i++;
			}
			return -1*i;
		}
		return 0;
	}
	
	public static void analysisFiles(File[] files, LineHandler handler){
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
				out.printf("!!!file read failed: %s %n", e);
			} finally {
				if (it != null)
					LineIterator.closeQuietly(it);
			}
		}// for file
		
		out.printf(Locale.ENGLISH, "analysisFiles: all(%d) sucess(%d) %n"
				, all, suc);
	}
	
	public static void analysisFiles(File[] files, LineHandler handler, int passes){
		
		for (int pass = 0; pass < passes; pass++) {
			out.printf(Locale.ENGLISH, "pass %d: %n", pass);
			Utils.analysisFiles(files, handler);
		}// for pass
	}
	
	private static Configuration conf = HadoopConfUtil.newConf(); 
	private static CompressionCodecFactory factory = new CompressionCodecFactory(conf); 
	private static FileSystem hdfs;
	static {
		try {
			hdfs = HadoopConfUtil.getFileSystem(null, null);
		} catch (IOException e) {
			e.printStackTrace();
			out.println(e);
		}
	}
	
	private final static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
	private final static String table = "/hive/warehouse/fig.db/fig_app_user/dt=";
	
	public static RemoteIterator<LocatedFileStatus> listHdfsFiles(LocalDate ld) {
		String dayStr = ld.format(formatter);
		RemoteIterator<LocatedFileStatus> lfss;
		try {
			lfss = hdfs.listFiles(
					new Path(table + dayStr + "/user_type=account" )
					, true);
		} catch (IllegalArgumentException | IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			out.println("hdfs.listFIles failed! dayStr: " + dayStr);
			return null;
		}
		return lfss;
	}
	
	public static RemoteIterator<LocatedFileStatus> listHdfsFiles(LocalDate ld, String tableName) {
		String dayStr = ld.format(formatter);
		String midTable = "/hive/warehouse/fig.db/" + tableName + "/dt=";
		RemoteIterator<LocatedFileStatus> lfss;
		try {
			lfss = hdfs.listFiles(
					new Path(midTable + dayStr + "/user_type=account" )
					, true);
		} catch (IllegalArgumentException | IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			out.println("hdfs.listFIles failed! dayStr: " + dayStr);
			return null;
		}
		return lfss;
	}
	
	public static void analysisHdfsFiles(RemoteIterator<LocatedFileStatus> lfss, LineHandler handler){

		int all = 0;
		int suc = 0;
		try {
			while(lfss.hasNext()){
				LocatedFileStatus lfs = lfss.next();
				CompressionCodec codec = factory.getCodec(lfs.getPath() ); 
				FSDataInputStream in = hdfs.open(lfs.getPath() );
				BufferedReader br;
				if(codec == null){
					br = new BufferedReader(new InputStreamReader(in));
				} else {
					CompressionInputStream comInputStream = codec.createInputStream(in);  
			        br = new BufferedReader(new InputStreamReader(comInputStream));
				}
				
				String line;
			    while ((line = br.readLine()) != null) {
			    	all++;
			    	if(handler.handle(line))
			    		suc++;
			    }
			    br.close();
			    in.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
			out.println(e);
		}// for file
		
		out.printf(Locale.ENGLISH, "analysisHdfsFiles finish: all(%d) sucess(%d) %n"
				, all, suc);
	}
	
	public static void main(String[] args) {
		int i = 10;
		out.printf("get default: %s %n",  1);
	}
}
