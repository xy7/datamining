package com.seasun.data.datamining;

import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.commons.io.Charsets;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class ReadHiveCZSG{
	private static String APPID = "1024appid";

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
		
		lr = new OnlineLogisticRegression(3, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(1e-1);// 1e-3
		lr.alpha(1 - 1.0e-5);// 学习率的指数衰减率,步长
	}

	public static void main(String[] args) throws Exception {
		
		Utils.loadConfigFile("./config.properties.hive");
		APPID = Utils.getOrDefault("appid", APPID);

		LocalDate start = LocalDate.parse(Utils.getOrDefault("train_start", "2015-11-01") );
		LocalDate end   = LocalDate.parse(Utils.getOrDefault("train_end", "2015-11-01") );
		int trainPass = Utils.getOrDefault("train_pass", 1);
		for(int i=0;i<trainPass;i++){
			out.printf(Locale.ENGLISH, "--------pass: %2d ---------%n", i);
			for(LocalDate ld=start; !ld.isAfter(end); ld=ld.plusDays(1) ){	
				train(ld);
			}
		}
		
		Utils.saveModel(lr);
		
		LocalDate evalStart = LocalDate.parse(Utils.getOrDefault("eval_start", "2015-11-01") );
		LocalDate evalEnd   = LocalDate.parse(Utils.getOrDefault("eval_end", "2015-11-01") );
		
		for(LocalDate ld=evalStart; !ld.isAfter(evalEnd); ld=ld.plusDays(1) ){	
			eval(ld);
		}
		
		Utils.printEvalRes(resMap);
	}
	
	private static Map<String, double[]> resMap = new HashMap<>();
	
	private static void eval(LocalDate ld){
		out.println("eval: " + ld.toString());
		getTargetValue(ld);
		
		RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld);
		if(lfss == null)
			return;
		
		Auc collector = new Auc();
		Integer[] res = {0, 0, 0, 0, 0, 0};//abcd;
		Map<String, Integer> lostLd = lost.get(ld);
		Utils.analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
				String accountId;
		        Vector input = new RandomAccessSparseVector(numFeatures);
		        try {
		        	accountId = parseLine(line, input, ld);
		        	if(accountId == null)
		        		return false;
		        	int targetValue = lostLd.get(accountId);
		        	
		        	Vector score = lr.classify(input);	
		        	//collector.add(targetValue, score);
		        	
		        	double s1 = score.get(0);
		        	double s2 = score.get(1);
		        	double s0 = 1 - s1 - s2;
		        	
					int predictValue = 0;
					if(s1 >= s2 && s1 >= s0)
						predictValue = 1;
					if(s2 > s1 && s2 > s0)
						predictValue = 2;
					
					if (targetValue == 1) {
						if (predictValue == 1) {
							res[0]++; //将流失用户预测正确
						} else {
							res[2]++; //将流失用户预测错误
						}
					} else if(targetValue == 2){
						if (predictValue == 2) {
							res[1]++; //留存用户预测正确
						} else {
							res[3]++; //留存用户预测错误
						}
					} else{
						if (predictValue == 0) {
							res[4]++; //流失用户预测正确
						} else {
							res[5]++; //流失用户预测错误
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
		
		int all = res[0] + res[1] + res[2] + res[3] + res[4] + res[5];
		out.printf("result matrix: lostcnt:%d	remaincnt:%d%n", res[0]+res[2], res[1]+res[3]);
		out.printf("A:%2.4f	B:%2.4f %n", (double)res[0]/all, (double)res[1]/all);
		out.printf("C:%2.4f	D:%2.4f %n", (double)res[2]/all, (double)res[3]/all);
		out.printf("E:%2.4f	F:%2.4f %n", (double)res[4]/all, (double)res[5]/all);

		double coverRate = (double) res[0]/(res[0]+res[2]);//覆盖率
		double rightRate = (double) (res[0]+res[3])/all;//正确率
		double hitRate = (double) res[0]/(res[0]+res[1]);//命中率
		out.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
				, coverRate, rightRate, hitRate);
		
		//out.printf(Locale.ENGLISH, "AUC = %.2f%n", collector.auc());
		
		double[] tmp = {coverRate, rightRate, hitRate};
		resMap.put(ld.toString(), tmp);
	}

	private static void train(LocalDate ld){
		out.println("train: " + ld.toString());
		getTargetValue(ld);
		RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld);
		if(lfss == null)
			return;
		
		Map<String, Integer> lostLd = lost.get(ld);
		int[] res = new int[3];//parse error, get target error, sucess
		rowstat = new int[]{0, 0, 0, 0, 0};
		Utils.analysisHdfsFiles(lfss, new LineHandler(){
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
	        	
	        	if (Utils.scores) {
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
		
		out.printf("columns size too small or appid not %s: %d, first_login_date error: %d, uptodate < 14: %d, last14LoginDaycnt not in[2,13]: %d  %n"
				, APPID, rowstat[0], rowstat[1], rowstat[2], rowstat[3]);
		out.printf("parse error: %d, get target error: %d, sucess: %d %n", res[0], res[1], res[2]);
	}


	private static Map<String, String> lineSplit(String line){

		String[] cols = line.split("\1");
		String appId = cols[0];
		if(!appId.equals(APPID) )
			return null;
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
		RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld.plusDays(14));
		if(lfss == null)
			return;
		
		Map<String, Integer> lostLd = new HashMap<>();
		Utils.analysisHdfsFiles(lfss, new LineHandler(){
			@Override
			public boolean handle(String line) {
				try{
					Map<String, String> cols = lineSplit(line);
					if(cols == null)
						return true;
					//out.println("line column size: " + cols.size() + "  values: " + cols);
			    	String accountId = cols.get("account_id");

			    	int last14LoginDaycnt = Integer.parseInt(cols.get("last14_login_daycnt") );
					int last7LoginDaycnt = Integer.parseInt(cols.get("last7_login_daycnt") );
					int next7LoginDaycnt = last14LoginDaycnt - last7LoginDaycnt;
					
					int targetValue = 0;//流失用户
					if(next7LoginDaycnt > 0 && last7LoginDaycnt > 0){//留存用户
						targetValue = 2;
					} else if(next7LoginDaycnt > 0 && last7LoginDaycnt == 0){//将流失用户
						targetValue = 1;
					}

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
	
	//返回帐号id
	private static int[] rowstat;
	private static String parseLine(String line, Vector featureVector, LocalDate ld){
		
		featureVector.setQuick(0, 1.0);//填充常量 k0
		
		Map<String, String> cols = lineSplit(line);
		
		if(cols == null || cols.size() < 60){
			//out.println("parse error, columns size to small: " + cols.size());
			rowstat[0]++;
			return null;
		}

		int i = 1;
		for(String k:trainIndex){
			String s = cols.get(k);
			if(s.equals("\\N"))//null值替换为0
				s = "0";
			featureVector.setQuick(i, Double.parseDouble(s));
			i++;
		}
		
		String fistLoginDate = cols.get("first_login_date");
		if(fistLoginDate == null || fistLoginDate.length() != 10){
			//out.println("fistLoginDate format error: " + fistLoginDate);
			rowstat[1]++;
			return null;
		}
		LocalDate firstLoginDate = LocalDate.parse(fistLoginDate);
		int uptodate = Utils.dateDiff(ld, firstLoginDate);
		if(uptodate < 14){
			//out.println("uptodate < 14: " + firstLoginDate);
			rowstat[2]++;
			return null;
		}
		
		int last14LoginDaycnt = Integer.parseInt( cols.get("last14_login_daycnt") );
		if(last14LoginDaycnt < 2 || last14LoginDaycnt > 13){
			//out.println("last7LoginDaycnt < 1");
			rowstat[3]++;
			return null;
		}
	
		featureVector.setQuick(i, uptodate);
		
		String accountId = cols.get("account_id");

		return accountId;
	}

}
