package com.seasun.data.datamining;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
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
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

/*
 * 根据空间自相似来预测
 */
public class ReadHiveSelfSimilar {
	private static int TARGET_AFTTER_DAYS = 14;

	private static String APPID = "100001";

	private static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);

	private static int numFeatures = 14;
	private static int SCORE_FREQ;
	private static double LOST_THRESHOLD = 1.0;

	public static Map<LocalDate, Map<String, Map<String, Integer>>> ldAccountMaps = new HashMap<>();
	
	private static final String[] names = {"online_dur", "login_cnt", "recharge_cnt", "recharge" };

	public static void main(String[] args) throws Exception {

		Utils.loadConfigFile("./config.properties.hive");
		APPID = Utils.getOrDefault("appid", APPID);
		SCORE_FREQ = Utils.getOrDefault("score_freq", 0);

		TARGET_AFTTER_DAYS = Utils.getOrDefault("target_after_days", 14);
		numFeatures = Utils.getOrDefault("num_features", 14);
		LOST_THRESHOLD = Utils.getOrDefault("lost_threshold", 1.0);
		
		OnlineLogisticRegression lr;
		lr = new OnlineLogisticRegression(2, 2 + names.length*numFeatures, new L1());
		lr.lambda(1e-4);// 先验分布的加权因子
		lr.learningRate(1e-1);// 1e-3
		lr.alpha(1 - 1.0e-5);// 学习率的指数衰减率,步长

		LocalDate start = LocalDate.parse(Utils.getOrDefault("train_start", "2015-11-01"));
		LocalDate end = LocalDate.parse(Utils.getOrDefault("train_end", "2015-11-01"));

		LocalDate evalStart = LocalDate.parse(Utils.getOrDefault("eval_start", "2015-11-01"));
		LocalDate evalEnd = LocalDate.parse(Utils.getOrDefault("eval_end", "2015-11-01"));

		// new code
		// step1/3, load all of hive data to map, write to local file
		// if exits local file, load to map
		loadAllHiveData(start, evalEnd.plusDays(numFeatures + TARGET_AFTTER_DAYS - 1));

		// step2/3, train data
		Map<LocalDate, Map<String, Vector>> samples = mapTransfer(start, end, numFeatures, true);
		out.println("train");
		Map<LocalDate, Map<String, Integer>> accountTargetValue = getTargetValue(start, end, samples, false);
		Map<Integer, List<Vector>> samplesClass = getClassValue(accountTargetValue, samples);
		train(lr, samples, accountTargetValue);
//		similarAnalysis(samplesClass);
		evalLR(lr, samples, accountTargetValue);

		// step3/3, eval data
		// eval 需要增加剔除过滤的逻辑
//		out.println("eval");
//		Map<LocalDate, Map<String, Vector>> evalSamples = mapTransfer(evalStart, evalEnd, numFeatures, true);
//		Map<LocalDate, Map<String, Integer>> accountTargetValue2 = getTargetValue(evalStart, evalEnd, evalSamples, false);
//		eval(accountTargetValue2, evalSamples, samplesClass);

	}
	
	private static void train(OnlineLogisticRegression lr
			, Map<LocalDate, Map<String, Vector>> samples
			, Map<LocalDate, Map<String, Integer>> accountTargetValue){
		out.println("train: ");
		
		for(Map.Entry<LocalDate, Map<String, Vector>> e: samples.entrySet()){
			LocalDate ld = e.getKey();
	
			for(Map.Entry<String, Vector> eInner:e.getValue().entrySet()){
				String accountId = eInner.getKey();
				Vector input = eInner.getValue();
				int targetValue = accountTargetValue.get(ld).getOrDefault(accountId, 0);
				lr.train(targetValue == 2? 1:0, input);
			}
		}
	}
	
	private static void evalLR(OnlineLogisticRegression lr
			, Map<LocalDate, Map<String, Vector>> samples
			, Map<LocalDate, Map<String, Integer>> accountTargetValue){
		out.println("evalLR: " );

		Auc collector = new Auc();
		Integer[] res = {0, 0, 0, 0};//abcd;
		for(Map.Entry<LocalDate, Map<String, Vector>> e: samples.entrySet()){
			LocalDate ld = e.getKey();
	
			for(Map.Entry<String, Vector> eInner:e.getValue().entrySet()){
				String accountId = eInner.getKey();
				Vector input = eInner.getValue();
				int targetValue = accountTargetValue.get(ld).getOrDefault(accountId, 0);
				if(targetValue == 2)
					targetValue = 1;
				else 
					targetValue = 0;
				
				double score = lr.classifyScalar(input);	
	        	collector.add(targetValue, score);
	        	
				int predictValue = score > Utils.CLASSIFY_VALUE ? 1 : 0;
				
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
	
			}
		}
		
		
		int all = res[0] + res[1] + res[2] + res[3];
		out.printf("result matrix: lostcnt:%d	remaincnt:%d%n", res[0]+res[2], res[1]+res[3]);
		out.printf("A:%2.4f	B:%2.4f %n", (double)res[0]/all, (double)res[1]/all);
		out.printf("C:%2.4f	D:%2.4f %n", (double)res[2]/all, (double)res[3]/all);

		double coverRate = (double) res[0]/(res[0]+res[2]);//覆盖率
		double rightRate = (double) (res[0]+res[3])/all;//正确率
		double hitRate = (double) res[0]/(res[0]+res[1]);//命中率
		out.printf(Locale.ENGLISH, "cover rate:%2.4f   right rate:%2.4f   hit rate:%2.4f  %n"
				, coverRate, rightRate, hitRate);
		
		out.printf(Locale.ENGLISH, "AUC = %.2f%n", collector.auc());
		
	}
	
	public static void similarAnalysis(Map<Integer, List<Vector>> samplesClass){

		double[][] mins = new double[3][3];
		double[][] avgs = new double[3][3];
		double[][] maxs = new double[3][3];
		for(int i=0;i<3;i++){
			for(int j=i;j<3;j++){
				double[] res = vectorsCompare(samplesClass.get(i), samplesClass.get(j));
				mins[i][j] = res[0];
				avgs[i][j] = res[1];
				maxs[i][j] = res[2];
			}
		}
		out.println("mins:");
		printArray(mins);
		out.println("avgs:");
		printArray(avgs);
		out.println("maxs:");
		printArray(maxs);
	}
	
	public static void printArray(double[][] mins){
		for(int i=0;i<mins.length;i++){
			for(int j=0;j<mins[0].length;j++){
				out.printf("%f\t", mins[i][j]);
			}
			out.printf("%n");
		}
	}
	
	public static double[] vectorsCompare(List<Vector> listA, List<Vector> listB){
		double sA[] = compu(listA);
		double sB[] = compu(listB);
		
		double min = 1.0, avg = 0.0, max = 0.0;
		int sizeA = listA.size();
		int sizeB = listB.size();
		int sizeAB = sizeA*sizeB;
		
		for(int a=0;a<sizeA;a++){
			Vector va = listA.get(a);
			double sa = sA[a];
			for(int b=0;b<sizeB;b++){
				Vector vb = listB.get(b);
				double sb = sB[b];
				double sMax = sa + sb;
				double similar = vectorSimilar(va, vb)/sMax;
				avg +=  similar/sizeAB;
				if(similar > max){
					max = similar;
					out.printf("get max: %f, va:%s, vb:%s %n", max, va.toString(), vb.toString());
				}
				if(similar < min)
					min = similar;
			}
		}
		out.printf("min:%f, avg:%f, max:%f %n", min, avg, max);
		double[] res = {min, avg, max};
		return res;
	}
	
	public static double[] compu(List<Vector> list){
		int size = list.size();
		double s[] = new double[size];
		for(int a=0;a<size;a++){
			Vector va = list.get(a);
			s[a] = va.dot(va);
		}
		return s;
	}

	public static void eval2(Map<LocalDate, Map<String, Integer>> accountTargetValue
			, Map<LocalDate, Map<String, Vector>> samples
			, Map<Integer, List<Vector>> samplesClass) {
		
		Map<Integer, Vector> avgSamplesClass = new HashMap<>(3);
		for (Map.Entry<Integer, List<Vector>> e : samplesClass.entrySet()) {
			Vector sum = new SequentialAccessSparseVector(numFeatures);
			for (Vector v : e.getValue()) {
				sum = sum.plus(v);
			}
			Vector avg = sum.divide(e.getValue().size());
			avgSamplesClass.put(e.getKey(), avg);
		}
		
		out.printf("avgSamplesClass: %s %n", avgSamplesClass.toString());
		
		int sampleCnt = 0;
		Integer[][] res = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };// abcd;

		for (Map.Entry<LocalDate, Map<String, Vector>> e : samples.entrySet()) {
			LocalDate ld = e.getKey();
			Map<String, Vector> map = e.getValue();
			for (Map.Entry<String, Vector> eInner : map.entrySet()) {
				String accountId = eInner.getKey();
				Vector input = eInner.getValue();
				int targetValue = accountTargetValue.get(ld).getOrDefault(accountId, 0);
				if (targetValue < 0 || targetValue > 2)
					continue;
				double[] sr = { 0.0, 0.0, 0.0 };// 平均相似度
				for(int t=0;t<3;t++){
					sr[t] = vectorSimilar(input, avgSamplesClass.get(t));
				}
				int predictValue = 1;
				if (sr[0] < sr[1] && sr[0] < sr[2])
					predictValue = 0;
				else if (sr[2] < sr[0] && sr[2] < sr[1])
					predictValue = 2;

				res[targetValue][predictValue]++;

				if (SCORE_FREQ != 0 && (++sampleCnt) % SCORE_FREQ == 0) {
					out.printf(
							"account:%s, input:%s, target:%d, predict:%d, similar:%f\t%f\t%f %n"
							, accountId, input.toString(), targetValue, predictValue
							, sr[0], sr[1], sr[2]);
				}
			}
		}
		
		Utils.printResMatrix(res);
	}

	public static Map<LocalDate, Map<String, Vector>> mapTransfer(LocalDate start, LocalDate end
			, int numFeatures, boolean filter) {

		Map<LocalDate, Map<String, Vector>> ldAccountVec = new HashMap<>();

		Map<LocalDate, Map<String, Integer>> lowLevelAccountIds = new HashMap<>();

		for (LocalDate ld = start; !ld.isAfter(end.plusDays(numFeatures - 1)); ld = ld.plusDays(1)) {

			Map<String, Map<String, Integer>> accountMaps = ldAccountMaps.get(ld);
			if (accountMaps == null) {
				out.println("ldAccountMaps get null ld: " + ld + "  map:" + ldAccountMaps);
				return null;
			}
			for (Map.Entry<String, Map<String, Integer>> e : accountMaps.entrySet()) {
				String accountId = e.getKey();

				Map<String, Integer> map = e.getValue();

				int roleLevel = map.getOrDefault("role_level", 0);
				if (roleLevel < 25) {// 后续需要过滤掉
					Map<String, Integer> lowLevel = lowLevelAccountIds.getOrDefault(ld, new HashMap<>());
					lowLevel.put(accountId, roleLevel);
					lowLevelAccountIds.put(ld, lowLevel);
				}

				LocalDate beforLd = ld.minusDays(numFeatures - 1);
				LocalDate beforSet = start.isAfter(beforLd) ? start : beforLd;

				
				
				for(int i=0;i<names.length;i++){
					int value = map.getOrDefault(names[i], 0);
					for (LocalDate ldCur = beforSet; !ldCur.isAfter(ld); ldCur = ldCur.plusDays(1)) {
						if (ldCur.isAfter(end))// 超过end日期向量不全，舍弃
							break;
						Map<String, Vector> accountVec = ldAccountVec.getOrDefault(ldCur, new HashMap<>());
						Vector v = accountVec.getOrDefault(accountId, new SequentialAccessSparseVector(numFeatures));
						v.set(0, 1);
						v.set(1, roleLevel);
						int index = Utils.dateDiff(ld, ldCur);
						v.setQuick(2 + i*numFeatures + index, value);
						accountVec.put(accountId, v);
						ldAccountVec.put(ldCur, accountVec);
					}
				}
				

			}

		}

		if (filter)
			filterMap(numFeatures, ldAccountVec, lowLevelAccountIds);

		return ldAccountVec;
	}

	public static void filterMap(int numFeatures, Map<LocalDate, Map<String, Vector>> ldAccountVec,
			Map<LocalDate, Map<String, Integer>> lowLevelAccountIds) {
		// 移除级别低的用户向量
		int lowLevelCnt = 0;
		for (Map.Entry<LocalDate, Map<String, Integer>> e : lowLevelAccountIds.entrySet()) {
			for (String s : e.getValue().keySet()) {
				for (int i = 0; i < numFeatures; i++) {
					LocalDate ldCur = e.getKey().plusDays(i);
					if (!ldAccountVec.containsKey(ldCur)) {
						// 越界，略过
						continue;
					}

					if (ldAccountVec.get(ldCur).remove(s) != null) {
						lowLevelCnt++;
					}
				}
			}
		}

		// 移除登陆天次不够的用户向量
		int noZeroCnt = 0;
		int zeroCnt = 0;
		Iterator<Map.Entry<LocalDate, Map<String, Vector>>> it = ldAccountVec.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<LocalDate, Map<String, Vector>> e = it.next();
			LocalDate ld = e.getKey();
			Iterator<Map.Entry<String, Vector>> itInner = e.getValue().entrySet().iterator();
			while (itInner.hasNext()) {
				Map.Entry<String, Vector> eInner = itInner.next();
				Vector v = eInner.getValue();
				int n = v.getNumNonZeroElements();
				if (n < 2) {
					zeroCnt++;
					itInner.remove();
				} else {
					noZeroCnt++;
				}
			}
		}

		out.printf("lowLevelCnt: %d, noZeroCnt: %d, zeroCnt: %d %n", lowLevelCnt, noZeroCnt, zeroCnt);
	}

	private static void loadHiveData(LocalDate ld) throws FileNotFoundException, IOException, ClassNotFoundException {
		if (ldAccountMaps.containsKey(ld)) {
			out.println("ldAccountMaps already exits: " + ld.toString());
			return;
		}

		String serialFileName = "./data/serial_data_mid_" + APPID + "_" + ld.toString() + ".out";
		File file = new File(serialFileName);
		if (!file.exists()) {
			RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld, "fig_app_user_day");
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

					if (accountId == null) {
						res[1]++;
						return false;
					}

					if (cols.size() < 6) {
						res[0]++;
						return false;
					}

					res[2]++;
					accountMaps.put(accountId, cols);

					return true;
				}
			});

			ldAccountMaps.put(ld, accountMaps);

			out.printf(
					"columns size too small: %d, appid != %s: %d, sucess: %d  %n"
					, res[0], APPID, res[1], res[2]);

			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
			out.writeObject(accountMaps);
			out.close();
		} else {
			out.println("serial file exists read to Map");
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
			Map<String, Map<String, Integer>> accountMap = (Map<String, Map<String, Integer>>) in.readObject();
			out.printf("date:%s size:%d %n", ld.toString(), accountMap.size());
			ldAccountMaps.put(ld, accountMap);
			in.close();
		}
	}

	// 1, load all of hive data to map, write to local file
	private static void loadAllHiveData(LocalDate ld1, LocalDate ld2) {
		for (LocalDate ld = ld1; !ld.isAfter(ld2); ld = ld.plusDays(1)) {
			out.println("load hive data: " + ld.toString());
			try {
				loadHiveData(ld);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	private static double vectorSimilar(Vector eval, Vector sample) {
		return eval.minus(sample).norm(2);
//		Vector diff = eval.minus(sample);
//		return diff.dot(diff);
	}

	private static void eval(Map<LocalDate, Map<String, Integer>> accountTargetValue
			, Map<LocalDate, Map<String, Vector>> samples
			, Map<Integer, List<Vector>> samplesClass) {
		out.println("eval start: ");
		// Map<LocalDate, Map<String, Integer>> accountTargetValue =
		// getTargetValue(start, end, samples);

		int sampleCnt = 0;
		Integer[][] res = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };// abcd;

		for (Map.Entry<LocalDate, Map<String, Vector>> e : samples.entrySet()) {
			LocalDate ld = e.getKey();
			Map<String, Vector> map = e.getValue();
			for (Map.Entry<String, Vector> eInner : map.entrySet()) {
				String accountId = eInner.getKey();
				Vector input = eInner.getValue();
				int targetValue = accountTargetValue.get(ld).getOrDefault(accountId, 0);
				if (targetValue < 0 || targetValue > 2)
					continue;

				double[] similar = { 0.0, 0.0, 0.0 };// 绝对相似度
				for (int i = 0; i <= 2; i++) {
					for (Vector v : samplesClass.get(i)) {
						similar[i] += vectorSimilar(input, v);
					}
				}

				double[] sr = { 0.0, 0.0, 0.0 };// 平均相似度
				for (int i = 0; i <= 2; i++)
					sr[i] = similar[i] / samplesClass.get(i).size();

				int predictValue = 1;
				if (sr[0] < sr[1] && sr[0] < sr[2])
					predictValue = 0;
				else if (sr[2] < sr[0] && sr[2] < sr[1])
					predictValue = 2;

				res[targetValue][predictValue]++;

				if (SCORE_FREQ != 0 && (++sampleCnt) % SCORE_FREQ == 0) {
					out.printf(
							"account:%s, input:%s, target:%d, predict:%d, similar:%f\t%f\t%f, avg similar:%f\t%f\t%f %n"
							, accountId, input.toString(), targetValue, predictValue
							, similar[0], similar[1], similar[2]
							, sr[0], sr[1], sr[2]);
				}
			}
		}

		int all = 0;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				all += res[i][j];

		out.printf("result matrix all: %d %n", all);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				out.printf("%2.4f \t", (double) res[i][j] / all);
			}
			out.printf("%n");
		}

	}
	
	

	private static Map<Integer, List<Vector>> getClassValue(Map<LocalDate, Map<String, Integer>> accountTargetValue
			, Map<LocalDate, Map<String, Vector>> samples) {
		out.println("train: ");
		Map<Integer, List<Vector>> samplesClass = new HashMap<>();
		for (int i = 0; i < 3; i++)
			samplesClass.put(i, new LinkedList<>());

		int[] sampleStat = { 0, 0, 0 };

		for (Map.Entry<LocalDate, Map<String, Vector>> e : samples.entrySet()) {
			LocalDate ld = e.getKey();
			Map<String, Vector> map = e.getValue();
			for (Map.Entry<String, Vector> eInner : map.entrySet()) {
				String accountId = eInner.getKey();
				Vector input = eInner.getValue();
				int targetValue = accountTargetValue.get(ld).getOrDefault(accountId, 0);
				if (targetValue < 0 || targetValue > 2)
					continue;

				samplesClass.get(targetValue).add(input);
				sampleStat[targetValue]++;
			}

		}

		out.printf("getClassValue finish, lost cnt:%d, may cnt:%d, retain cnt: %d %n", sampleStat[0], sampleStat[1],
				sampleStat[2]);

		return samplesClass;
	}

	private static String lineSplit(String line, LocalDate ld, Map<String, Integer> res) {

		String[] cols = line.split("\1");
		String appId = cols[0];
		if (!appId.equals(APPID))
			return null;

		String accountId = cols[1];

		// res.put("app_id", cols[0]);
		// res.put("account_id", cols[1]);
		// res.put("device_id", cols[2]);
		// res.put("mac", cols[3]);

		String mapStr = cols[4];
		String role_level = cols[5];
		String login_cnt = cols[6];
		String online_dur = cols[7];
		String recharge_cnt = cols[8];
		String recharge = cols[9];

		Map<String, String> strMap = parse2map(mapStr);

		String fistLoginDate = strMap.get("first_login_date");
		// if (fistLoginDate == null || fistLoginDate.length() != 10) {
		// return null;
		// }
		LocalDate firstLoginDate = LocalDate.parse(fistLoginDate);
		int uptodate = Utils.dateDiff(ld, firstLoginDate) + 1;
		res.put("up_to_date", uptodate);

		// res.putAll(parse2map(mapStr));
		res.put("role_level", Str2Int(role_level));
		res.put("login_cnt", Str2Int(login_cnt));
		res.put("online_dur", Str2Int(online_dur));
		res.put("recharge_cnt", Str2Int(recharge_cnt));
		res.put("recharge", Str2Int(recharge));

		return accountId;
	}

	private static int Str2Int(String s) {
		if (s.equalsIgnoreCase("\\N") || s.isEmpty() || s.equals(" ")) {
			return 0;
		} else {
			return Integer.parseInt(s);
		}
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

	public static Map<LocalDate, Map<String, Integer>> getTargetValue(LocalDate start, LocalDate end
			, Map<LocalDate, Map<String, Vector>> samples, boolean filter) {

		Map<LocalDate, Map<String, Vector>> ldTarget = mapTransfer(start.plusDays(numFeatures)
				, end.plusDays(numFeatures), TARGET_AFTTER_DAYS, filter);

		Map<LocalDate, Map<String, Integer>> res = new HashMap<>(samples.size());
		int[] rowstat = { 0, 0, 0, 0, 0, 0 };
		for (Map.Entry<LocalDate, Map<String, Vector>> e : samples.entrySet()) {
			LocalDate ld = e.getKey();
			Map<String, Integer> targetClass = new HashMap<>();
			res.put(ld, targetClass);
			Map<String, Vector> target = ldTarget.get(ld.plusDays(numFeatures));
			if (target == null) {
				out.println("ldTarget get null ld: " + ld + "  ldTarget:" + ldTarget);
				return null;
			}
			Map<String, Vector> map = e.getValue();
			for (Map.Entry<String, Vector> eInner : map.entrySet()) {
				String accountId = eInner.getKey();
				if (!target.containsKey(accountId)) {
					targetClass.put(accountId, 0);
					rowstat[2]++;
				} else {
					Vector v = target.get(accountId);
					int nextHalfLoginDaycnt = 0;
					for (int i = 0; i < TARGET_AFTTER_DAYS / 2; i++) {
						if (v.get(i) > LOST_THRESHOLD) {
							nextHalfLoginDaycnt = 1;
							break;
						}
					}

					int lastHalfLoginDaycnt = 0;
					for (int i = TARGET_AFTTER_DAYS / 2; i < TARGET_AFTTER_DAYS; i++) {
						if (v.get(i) > LOST_THRESHOLD) {
							lastHalfLoginDaycnt = 1;
							break;
						}
					}

					int targetValue = 1;// 将流失用户
					if (nextHalfLoginDaycnt > 0 && lastHalfLoginDaycnt > 0) {// 留存用户
						targetValue = 2;
						rowstat[3]++;
					} else if (nextHalfLoginDaycnt == 0) {// 流失用户
						targetValue = 0;
						rowstat[4]++;
					} else {
						rowstat[5]++;
					}

					targetClass.put(accountId, targetValue);
				}

			}
		}

		out.printf(
				"get target null cnt: %d, get field exception cnt: %d, target not exits:%d, lost cnt:%d, may cnt:%d, retain cnt:%d %n"
				, rowstat[0], rowstat[1], rowstat[2], rowstat[4], rowstat[5], rowstat[3]);

		return res;
	}

}
