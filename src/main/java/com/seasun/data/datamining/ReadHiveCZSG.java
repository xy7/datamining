package com.seasun.data.datamining;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Serializable;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.commons.io.Charsets;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class ReadHiveCZSG {
	private static final int TARGET_AFTTER_DAYS = 30;

	private static int RETAIN_THRESHOLD = 15;

	private static String APPID = "1024appid";

	private static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);

	private static Map<LocalDate, Map<String, Integer>> lost = new HashMap<>();
	private static OnlineLogisticRegression lr;
	private static int numFeatures;

	private static List<String> trainIndex = null;
	static {
		trainIndex = Arrays.asList(new String[] { "role_level"
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
				, "last14_login_daycnt" });

		numFeatures = trainIndex.size() + 2;

		lr = new OnlineLogisticRegression(3, numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(1e-1);// 1e-3
		lr.alpha(1 - 1.0e-5);// 学习率的指数衰减率,步长
	}

	private static int SCORE_FREQ;

	private static Map<LocalDate, Map<String, Map<String, Integer>>> ldAccountMaps = new HashMap<>();
	
	public static void main(String[] args) throws Exception {

		Utils.loadConfigFile("./config.properties.hive");
		APPID = Utils.getOrDefault("appid", APPID);
		SCORE_FREQ = Utils.getOrDefault("score_freq", 0);
		MAY_LOST_REPEAT_CNT = Utils.getOrDefault("may_lost_repeat_cnt", 1);
		RETAIN_THRESHOLD = Utils.getOrDefault("retain_threshold", 15);

		LocalDate start = LocalDate.parse(Utils.getOrDefault("train_start", "2015-11-01"));
		LocalDate end = LocalDate.parse(Utils.getOrDefault("train_end", "2015-11-01"));

		LocalDate evalStart = LocalDate.parse(Utils.getOrDefault("eval_start", "2015-11-01"));
		LocalDate evalEnd = LocalDate.parse(Utils.getOrDefault("eval_end", "2015-11-01"));

		//new code 
		//step1/3, load all of hive data to map, write to local file
		//   if exits local file, load to map
		String serialFileName = "./serial_data_" + APPID + ".out";
		File file = new File(serialFileName);
		if (!file.exists()) {
			out.println("serial file not exists, now write to it");
			loadAllHiveData(start, end);
			loadAllHiveData(start.plusDays(TARGET_AFTTER_DAYS), end.plusDays(TARGET_AFTTER_DAYS));
			loadAllHiveData(evalStart, evalEnd);
			loadAllHiveData(evalStart.plusDays(TARGET_AFTTER_DAYS), evalEnd.plusDays(TARGET_AFTTER_DAYS));
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
			out.writeObject(ldAccountMaps);
			out.close();
		} else{
			out.println("serial file exists read to Map");
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
			ldAccountMaps = (Map<LocalDate, Map<String, Map<String, Integer>>>) in.readObject();
			in.close();
		}
		//step2/3, train data
		int trainPass = Utils.getOrDefault("train_pass", 1);
		for (int i = 0; i < trainPass; i++) {
			out.printf(Locale.ENGLISH, "--------pass: %3d ---------%n", i);
			for (LocalDate ld = start; !ld.isAfter(end); ld = ld.plusDays(1)) {
				train(ld);
			}
		}
		Utils.saveModel(lr);
		
		//step3/3, eval data
		out.println("eval sample");
		for (LocalDate ld = start; !ld.isAfter(end); ld = ld.plusDays(1)) {
			eval(ld);
		}
		out.println("eval new");
		for (LocalDate ld = evalStart; !ld.isAfter(evalEnd); ld = ld.plusDays(1)) {
			eval(ld);
		}
	}
	
	//1, load all of hive data to map, write to local file
	private static void loadAllHiveData(LocalDate ld1, LocalDate ld2){
		for (LocalDate ld = ld1; !ld.isAfter(ld2); ld = ld.plusDays(1)) {
			out.println("load hive data: " + ld.toString());
			if(ldAccountMaps.containsKey(ld))
				continue;
			
			RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld);
			if (lfss == null)
				return;
			
			int[] res = { 0, 0, 0 };
	
			Map<String, Map<String, Integer>> accountMaps = new HashMap<>();
			
			final LocalDate ldFinal = ld;
			Utils.analysisHdfsFiles(lfss, new LineHandler() {
				@Override
				public boolean handle(String line) {
	
					Map<String, Integer> cols = new HashMap<>();
					String accountId = lineSplit(line, ldFinal, cols);

					if(cols.size() < 50){
						res[0]++;
						return false;
					}
					
					if(accountId == null){
						res[1]++;
						return false;
					}

					res[2]++;
					accountMaps.put(accountId, cols);

					return true;
				}
			});
			
			ldAccountMaps.put(ld, accountMaps);

			out.printf(
					"columns size too small: %d, appid != %s or first_login_date error: %d, sucess: %d  %n"
					, res[0], APPID, res[1], res[2]);
		}
	}

	private static void eval(LocalDate ld) {
		out.println("eval: " + ld.toString());
		Integer[][] res = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };// abcd;
		
		Map<String, Map<String, Integer>> samples = ldAccountMaps.get(ld);
		for(Map.Entry<String, Map<String, Integer>> sample:samples.entrySet() ){
			String accountId = sample.getKey();
			Vector input = parseColumnMap(sample.getValue(), ld);
			if(input == null)
				continue;
				
			int targetValue = getTargetValue(ld, accountId);
			if(targetValue < 0)
				continue;
			
			//classify
			Vector score = lr.classify(input);

			double s1 = score.get(0);
			double s2 = score.get(1);
			double s0 = 1 - s1 - s2;

			int predictValue = 0;
			if(s1>=Utils.CLASSIFY_VALUE || (s1 >= s2 && s1 >= s0 )){
				predictValue = 1;
			} else if (s2 > s1 && s2 > s0)
				predictValue = 2;
			
			res[targetValue][predictValue] ++;
		}

		int all = 0;
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				all += res[i][j];
		
		out.printf("result matrix all: %d %n", all);
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				out.printf("%2.4f \t", (double) res[i][j] / all);
			}
			out.printf("%n");
		}
		
	}

	private static int MAY_LOST_REPEAT_CNT;
	private static void train(LocalDate ld) {
		out.println("train: " + ld.toString());
		int sampleCnt = 0;
		rowstat = new int[] { 0, 0, 0, 0, 0 };
		Map<String, Map<String, Integer>> samples = ldAccountMaps.get(ld);
		for(Map.Entry<String, Map<String, Integer>> sample:samples.entrySet() ){
			new RandomAccessSparseVector(numFeatures);
			String accountId = sample.getKey();
			Vector input = parseColumnMap(sample.getValue(), ld);
			if(input == null)
				continue;
				
			int targetValue = getTargetValue(ld, accountId);
			if(targetValue < 0)
				continue;
			
			//print score
			if (SCORE_FREQ != 0 && (++sampleCnt) % SCORE_FREQ == 0) {
				// check performance while this is still news
				double logP = lr.logLikelihood(targetValue, input);
				Vector vec = lr.classify(input);
				double p;
				if (targetValue >= 1)
					p = vec.get(targetValue - 1);
				else
					p = 1 - vec.get(0) - vec.get(1);
				out.printf(Locale.ENGLISH, "sampleCnt: %d  %2d  %1.4f  |  %2.6f %10.4f%n",
						sampleCnt, targetValue, p, lr.currentLearningRate(), logP);
			}

			// now update model
			// 将流失用户训练3次增加样本
			if (targetValue == 1) {
				for (int i = 0; i < MAY_LOST_REPEAT_CNT; i++)
					lr.train(targetValue, input);
			} else
				lr.train(targetValue, input);
		}
		
		out.printf(
				"train finish date:%s, first_login_date error: %d, uptodate < 14: %d, last14LoginDaycnt not in[2,13]: %d, getTargetValue error:%d   %n"
				,ld.toString(), rowstat[1], rowstat[2], rowstat[3], rowstat[0]);

	}

	private static String lineSplit(String line, LocalDate ld, Map<String, Integer> res) {

		String[] cols = line.split("\1");
		String appId = cols[0];
		if (!appId.equals(APPID))
			return null;
		
		String accountId = cols[1];

		//res.put("app_id", cols[0]);
		//res.put("account_id", cols[1]);
		//res.put("device_id", cols[2]);
		//res.put("mac", cols[3]);

		String mapStr = cols[4];
		String mapInt = cols[5];
		Map<String, String> strMap = parse2map(mapStr);
		
		String fistLoginDate = strMap.get("first_login_date");
		if (fistLoginDate == null || fistLoginDate.length() != 10) {
			// out.println("fistLoginDate format error: " + fistLoginDate);
			return null;
		}
		LocalDate firstLoginDate = LocalDate.parse(fistLoginDate);
		int uptodate = Utils.dateDiff(ld, firstLoginDate);
		res.put("up_to_date", uptodate);

		//res.putAll(parse2map(mapStr));
		res.putAll(parse2mapint(mapInt));

		return accountId;
	}

	private static Map<String, String> parse2map(String mapInt) {
		String[] intCols = mapInt.split("\2");
		Map<String, String> res = new HashMap<>();
		for (String col : intCols) {
			// out.println("167: " + col);
			String[] kv = col.split("\3");
			String value = "\\N";
			if (kv.length >= 2)
				value = kv[1];
			
			res.put(kv[0], value);
		}
		return res;
	}
	
	private static Map<String, Integer> parse2mapint(String mapInt) {
		Map<String, String> map = parse2map(mapInt);
		Map<String, Integer> res = new HashMap<>();
		for(String k:map.keySet()){
			if (map.get(k).equals("\\N"))// null值替换为0
				res.put(k, 0);
			else 
				res.put(k, Integer.parseInt(map.get(k)));
		}

		return res;
	}

	private static int[] rowstat;
	
	private static int getTargetValue(LocalDate ld, String accountId) {
		
		try {
			Map<String, Integer> cols = ldAccountMaps.get(ld.plusDays(TARGET_AFTTER_DAYS)).get(accountId);
			if (cols == null){
				rowstat[0]++;
				return -1;
			}
		

			int lastAllLoginDaycnt = cols.get("last30_login_daycnt");
			int lastHalfLoginDaycnt = cols.get("last14_login_daycnt");
			int nextHalfLoginDaycnt = lastAllLoginDaycnt - lastHalfLoginDaycnt;
			//int last30LoginDaycnt = Integer.parseInt(cols.get("last30_login_daycnt"));

			int targetValue = 1;// 将流失用户
			if (nextHalfLoginDaycnt > 0 && lastHalfLoginDaycnt > 0) {// 留存用户
				targetValue = 2;
			} else if (nextHalfLoginDaycnt == 0 ) {// 流失用户
				targetValue = 0;
			}

			return targetValue;
		} catch (Exception e) {
			// e.printStackTrace();
			//out.println(e);
			rowstat[0]++;
			return -2;
		}
	
	}

	// 返回帐号id
	private static Vector parseColumnMap(Map<String, Integer> cols, LocalDate ld) {

		Vector featureVector = new RandomAccessSparseVector(numFeatures);
		featureVector.setQuick(0, 1.0);// 填充常量 k0

		int i = 1;
		for (String k : trainIndex) {
			Integer s = cols.get(k);
//			if (s.equals("\\N"))// null值替换为0
//				s = "0";
			featureVector.setQuick(i, (double)s);
			i++;
		}

		int uptodate = cols.get("up_to_date");
		if (uptodate < 14) {
			// out.println("uptodate < 14: " + firstLoginDate);
			rowstat[2]++;
			return null;
		}

		int last14LoginDaycnt = cols.get("last14_login_daycnt");
		if (last14LoginDaycnt < 2 || last14LoginDaycnt > 13) {
			// out.println("last7LoginDaycnt < 1");
			rowstat[3]++;
			return null;
		}

		featureVector.setQuick(i, uptodate);

		return featureVector;
	}

}
