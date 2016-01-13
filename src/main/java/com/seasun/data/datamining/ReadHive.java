package com.seasun.data.datamining;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.commons.io.Charsets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.io.compress.CompressionInputStream;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.github.dataswitch.util.HadoopConfUtil;

public class ReadHive{
	private static final String APPID = "1024appid";
	private static Configuration conf = HadoopConfUtil.newConf(); 
	private static CompressionCodecFactory factory = new CompressionCodecFactory(conf); 
	private static FileSystem hdfs;
	private static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
	private static String table = "/hive/warehouse/fig.db/fig_app_user/dt=";
	
	private static PrintWriter output = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
	
	private static Map<String, Integer> lost = new HashMap<>();
	private static OnlineLogisticRegression lr;
	private static int numFeatures;
	
	private static List<String> trainIndex = null;
	static {
		trainIndex = Arrays.asList(new String[]{"role_level"
				, "last1_login_cnt"
				, "last2_login_cnt"
				, "last3_login_cnt"
				, "last4_login_cnt"
				, "last5_login_cnt"
				, "last6_login_cnt"
				, "last7_login_cnt"
				, "last14_login_cnt"
				, "last1_login_daycnt"
				, "last2_login_daycnt"
				, "last3_login_daycnt"
				, "last4_login_daycnt"
				, "last5_login_daycnt"
				, "last6_login_daycnt"
				, "last7_login_daycnt"
				, "last14_login_daycnt"});
		
		numFeatures = trainIndex.size() + 2;
	}
	
	static {
		try {
			hdfs = HadoopConfUtil.getFileSystem(null, null);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println(e);
		}
		
		lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(1e-1);// 1e-3
		lr.alpha(1 - 1.0e-5);// 学习率的指数衰减率,步长
	}

	public static void main(String[] args) throws Exception {
		
		if(!LogisticRegressionTrain.loadConfigFile("./config.properties")){
			if(!LogisticRegressionTrain.loadConfigFile("./config/config.properties")){
				System.out.println("load config file error");
				return;
			}
		}
	
		LocalDate start = LocalDate.parse("2015-08-01");
		LocalDate end   = LocalDate.parse("2015-08-01");
		for(int i=0;i<LogisticRegressionTrain.TARIN_PASSES;i++){
			output.printf(Locale.ENGLISH, "--------pass: %2d ---------%n", i);
			for(LocalDate ld=start; !ld.isAfter(end); ld=ld.plusDays(1) ){
				getTargetValue(ld);
				train(ld);
			}
		}
	}

	private static void train(LocalDate ld) throws FileNotFoundException, IOException {
		String dayStr = ld.format(formatter);
		RemoteIterator<LocatedFileStatus> lfss = hdfs.listFiles(
				new Path(table + dayStr + "/user_type=account" )
				, true);
		
		analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
				String accountId;
		        Vector input = new RandomAccessSparseVector(numFeatures);
		        try {
		        	accountId = parseLine(line, input, ld);
		        	if(accountId == null)
		        		return false;
		        	int targetValue = lost.get(accountId);
		        	
		        	if (LogisticRegressionTrain.scores) {
						// check performance while this is still news
						double logP = lr.logLikelihood(targetValue, input);
						double p = lr.classifyScalar(input);
						output.printf(Locale.ENGLISH, "%2d  %1.4f  |  %2.6f %10.4f%n",
								targetValue, p, lr.currentLearningRate(), logP);
					}

					// now update model
					lr.train(targetValue, input);
				} catch (Exception e) {
					e.printStackTrace();
					System.out.println(e);
					return false;
				}
		        return true;
			}
		});
	}

	private static Map<String, String> lineSplit(String line){

		String[] cols = line.split("\1");
		
		Map<String, String> res = new HashMap<>();
		res.put("app_id",		cols[0]);
		res.put("account_id",	cols[1]);
		res.put("device_id",	cols[2]);
		res.put("mac",			cols[3]);
		
		String mapStr		= cols[4];
		String mapInt		= cols[5];
		
		res.putAll(parse2map(mapStr));
		
		res.putAll(parse2map(mapInt));

		return res;
	}

	private static Map<String, String> parse2map(String mapInt) {
		String[] intCols = mapInt.split("\2");
		Map<String, String> res = new HashMap<>();
		for(String col:intCols){
			String[] kv = col.split("\3");
			res.put(kv[0], kv[1]);
		}
		return res;
	}
	
	private static void getTargetValue(LocalDate ld) throws Exception{
		lost.clear();
		String dayStr = ld.plusDays(14).format(formatter);
		RemoteIterator<LocatedFileStatus> lfss = hdfs.listFiles(
				new Path(table + dayStr + "/user_type=account" )
				, true);
		analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
				try{
					Map<String, String> cols = lineSplit(line);
					System.out.println("line column size: " + cols.size() + "  values: " + cols);
			    	String accountId = cols.get("account_id");
					
					String appId = cols.get("app_id");
					if(!appId.equals(APPID))
						return true;
					
					int last7LoginDaycnt = Integer.parseInt(cols.get("last7_login_daycnt") );
					int targetValue = last7LoginDaycnt>0?1:0;
					lost.put(accountId, targetValue);
					return true;
				} catch(Exception e) {
					e.printStackTrace();
					System.out.println(e);
					return false;
				}
			}
			
		});
	}
	
	private static void analysisHdfsFiles(RemoteIterator<LocatedFileStatus> lfss, LineHandler handler){
		analysisHdfsFiles(lfss, handler, 1);
	}
	
	private static void analysisHdfsFiles(RemoteIterator<LocatedFileStatus> lfss, LineHandler handler, int passes){
		for (int pass = 0; pass < passes; pass++) {
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
				System.out.println(e);
			}// for file
			
			output.printf(Locale.ENGLISH, "pass %d: all(%d) sucess(%d) %n"
					, pass, all, suc);
		}// for pass
	}
	
	public static int dayDiff(LocalDate day1, LocalDate day2){

		int i = 0;
		for(LocalDate ld=day1; !ld.isAfter(day2); ld=ld.plusDays(1)){
			i++;
		}
		return i;
	}

	public static String parseLine(String line, Vector featureVector, LocalDate ld) throws Exception {
		
		featureVector.setQuick(0, 1.0);//填充常量 k0
		
		Map<String, String> cols = lineSplit(line);
		if(cols.size() < 60)
			throw new Exception("parse error, columns size to small: " + cols.size());
		
		int i = 0;
		for(String k:trainIndex){
			String s = cols.get(k);
			if(s.equals("\\N"))//null值替换为空值
				s = "0";
			featureVector.setQuick(i + 1, Double.parseDouble(s));
			i++;
		}
		
		LocalDate firstLoginDate = LocalDate.parse(cols.get("first_login_date") );
		int uptodate = dayDiff(firstLoginDate, ld);
		featureVector.setQuick(i, uptodate);
		
		String accountId = cols.get("account_id");
		
		String appId = cols.get("app_id");
		if(!appId.equals(APPID))
			return null;
		
		return accountId;//返回帐号id
	}


}
