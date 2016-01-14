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
	
	private static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
	
	private static Map<LocalDate, Map<String, Integer>> lost = new HashMap<>();
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
	
		try {
			hdfs = HadoopConfUtil.getFileSystem(null, null);
		} catch (IOException e) {
			e.printStackTrace();
			out.println(e);
		}
		
		lr = new OnlineLogisticRegression(2, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(1e-1);// 1e-3
		lr.alpha(1 - 1.0e-5);// 学习率的指数衰减率,步长
	}

	public static void main(String[] args) throws Exception {
		
		if(!LogisticRegressionTrain.loadConfigFile("./config.properties")){
			if(!LogisticRegressionTrain.loadConfigFile("./config/config.properties")){
				out.println("load config file error");
				return;
			}
		}
	
		LocalDate start = LocalDate.parse("2015-11-01");
		LocalDate end   = LocalDate.parse("2015-11-01");
		for(int i=0;i<LogisticRegressionTrain.TARIN_PASSES;i++){
			out.printf(Locale.ENGLISH, "--------pass: %2d ---------%n", i);
			for(LocalDate ld=start; !ld.isAfter(end); ld=ld.plusDays(1) ){	
				train(ld);
			}
		}
		
		for(LocalDate ld=start; !ld.isAfter(end); ld=ld.plusDays(1) ){	
			eval(ld);
		}
	}
	
	private static Map<LocalDate, double[]> resMap = new HashMap<>();
	
	private static void eval(LocalDate ld){
		out.println("eval: " + ld.toString());
		getTargetValue(ld);
		
		RemoteIterator<LocatedFileStatus> lfss = listHdfsFiles(ld);
		if(lfss == null)
			return;
		
		Integer[] res = {0, 0, 0, 0};//abcd;
		Map<String, Integer> lostLd = lost.get(ld);
		analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
				String accountId;
		        Vector input = new RandomAccessSparseVector(numFeatures);
		        try {
		        	accountId = parseLine(line, input, ld);
		        	if(accountId == null)
		        		return false;
		        	int targetValue = lostLd.get(accountId);
		        	
		        	double score = lr.classifyScalar(input);
					int predictValue = score > LogisticRegressionTrain.CLASSIFY_VALUE ? 1 : 0;
					
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
				} catch (Exception e) {
					//e.printStackTrace();
					//out.println(e);
					return false;
				}
		        return true;
			}
		});
		
		int all = res[0] + res[1] + res[2] + res[3];
		out.printf("result matrix: lostcnt:%d	remaincnt:%d%n", res[0]+res[2], res[1]+res[3]);
		out.printf("A:%2.4f	B:%2.4f %n", (double)res[0]/all, (double)res[1]/all);
		out.printf("C:%2.4f	D:%2.4f %n", (double)res[2]/all, (double)res[3]/all);

		double coverRate = (double) res[0]/(res[0]+res[2]);//覆盖率
		double rightRate = (double) (res[0]+res[3])/all;//正确率
		double hitRate = (double) res[0]/(res[0]+res[1]);//命中率
		out.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
				, coverRate, rightRate, hitRate);
		
		double[] tmp = {coverRate, rightRate, hitRate};
		resMap.put(ld, tmp);
	}

	private static void train(LocalDate ld){
		out.println("train: " + ld.toString());
		getTargetValue(ld);
		RemoteIterator<LocatedFileStatus> lfss = listHdfsFiles(ld);
		if(lfss == null)
			return;
		
		Map<String, Integer> lostLd = lost.get(ld);
		int[] res = new int[3];//parse error, get target error, sucess
		rowstat = new int[]{0, 0, 0, 0};
		analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
		        Vector input = new RandomAccessSparseVector(numFeatures);
		        String accountId = parseLine(line, input, ld);
	        	if(accountId == null){
	        		res[0]++;
	        		return false;
	        	}
	        	
	        	int targetValue = lostLd.getOrDefault(accountId, -1);
	        	if(targetValue == -1){
	        		res[1]++;
	        		return false;
	        	}
	        	
	        	if (LogisticRegressionTrain.scores) {
					// check performance while this is still news
					double logP = lr.logLikelihood(targetValue, input);
					double p = lr.classifyScalar(input);
					out.printf(Locale.ENGLISH, "%2d  %1.4f  |  %2.6f %10.4f%n",
							targetValue, p, lr.currentLearningRate(), logP);
				}

				// now update model
				lr.train(targetValue, input);
				res[2]++;
		        return true;
			}
		});
		
		out.printf("columns size to small: %d, app_id: %d, last7LoginDaycnt < 1: %d, uptodate < 14: %d  %n"
				, rowstat[0], rowstat[1], rowstat[2], rowstat[3]);
		out.printf("parse error: %d, get target error: %d, sucess: %d %n", res[0], res[1], res[2]);
	}

	private static RemoteIterator<LocatedFileStatus> listHdfsFiles(LocalDate ld) {
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
			//out.println("167: " + col);
			String[] kv = col.split("\3");
			String value = "";
			if(kv.length >= 2)
				value = kv[1];
			
			res.put(kv[0], value);
		}
		return res;
	}
	
	private static void getTargetValue(LocalDate ld){
		out.println("getTargetValue: " + ld.toString());
		if(lost.containsKey(ld))
			return;
		RemoteIterator<LocatedFileStatus> lfss = listHdfsFiles(ld.plusDays(14));
		if(lfss == null)
			return;
		
		Map<String, Integer> lostLd = new HashMap<>();
		analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
				try{
					Map<String, String> cols = lineSplit(line);
					//out.println("line column size: " + cols.size() + "  values: " + cols);
			    	String accountId = cols.get("account_id");
					
					String appId = cols.get("app_id");
					if(!appId.equals(APPID))
						return true;
					
					int last7LoginDaycnt = Integer.parseInt(cols.get("last7_login_daycnt") );
					int targetValue = last7LoginDaycnt>0?1:0;
					lostLd.put(accountId, targetValue);
					return true;
				} catch(NullPointerException e) {
					//e.printStackTrace();
					//out.println(e);
					return false;
				}
			}
			
		});
		
		lost.put(ld, lostLd);
	}
	
	
	private static void analysisHdfsFiles(RemoteIterator<LocatedFileStatus> lfss, LineHandler handler){

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
	
	//day1 - day2 + 1
	public static int dateDiff(LocalDate day1, LocalDate day2){
		if(day1.equals(day2)){
			return 0;
		} else if(day1.isAfter(day2) ){
			int i = 0;
			for(LocalDate ld=day2; !ld.isAfter(day1); ld=ld.plusDays(1)){
				i++;
			}
			return -1*i;
		} else if(day1.isBefore(day2)) {
			int i = 0;
			for(LocalDate ld=day1; !ld.isAfter(day2); ld=ld.plusDays(1)){
				i++;
			}
			return i;
		}
		return 0;
	}

	//返回帐号id
	private static int[] rowstat;
	private static String parseLine(String line, Vector featureVector, LocalDate ld){
		
		featureVector.setQuick(0, 1.0);//填充常量 k0
		
		Map<String, String> cols;
	
		cols = lineSplit(line);
		
		if(cols == null || cols.size() < 60){
			//out.println("parse error, columns size to small: " + cols.size());
			rowstat[0]++;
			return null;
		}
		
		String appId = cols.get("app_id");
		if(!appId.equals(APPID)){
			rowstat[1]++;
			return null;
		}
	
		int i = 0;
		for(String k:trainIndex){
			String s = cols.get(k);
			if(s.equals("\\N"))//null值替换为空值
				s = "0";
			featureVector.setQuick(i + 1, Double.parseDouble(s));
			i++;
		}
		
		String fistLoginDate = cols.get("first_login_date");
		if(fistLoginDate == null || fistLoginDate.length() != 10){
			rowstat[2]++;
			return null;
		}
		LocalDate firstLoginDate = LocalDate.parse(fistLoginDate);
		
		//+ "and first_login_date <= '"+ ld.minusDays(13).format(formatter) +"' \n"
		//+ "and last7_login_daycnt >= 1\n"
		
		int last7LoginDaycnt = Integer.parseInt( cols.get("last7_login_daycnt") );
		if(last7LoginDaycnt < 1){
			//out.println("last7LoginDaycnt < 1");
			rowstat[2]++;
			return null;
		}
		int uptodate = dateDiff(ld, firstLoginDate);
		if(uptodate < 14){
			//out.println("uptodate < 14");
			rowstat[3]++;
			return null;
		}
				
		featureVector.setQuick(i, uptodate);
		
		String accountId = cols.get("account_id");

		return accountId;
	}

}
