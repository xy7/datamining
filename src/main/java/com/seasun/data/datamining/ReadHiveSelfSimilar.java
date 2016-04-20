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
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.Charsets;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/*
 * 根据空间自相似来预测
 */
public class ReadHiveSelfSimilar {
	private static int TARGET_AFTTER_DAYS = 14;

	private static String APPID = "100001";

	private static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);

	private static int numFeatures;

	private static Map<LocalDate, Map<String, Map<String, Integer>>> ldAccountMaps = new HashMap<>();

	public static void main(String[] args) throws Exception {

		Utils.loadConfigFile("./config.properties.hive");
		APPID = Utils.getOrDefault("appid", APPID);

		TARGET_AFTTER_DAYS = Utils.getOrDefault("target_after_days", 14);
		numFeatures = Utils.getOrDefault("num_features", 14);

		LocalDate start = LocalDate.parse(Utils.getOrDefault("train_start", "2015-11-01"));

		LocalDate evalStart = LocalDate.parse(Utils.getOrDefault("eval_start", "2015-11-01"));

		// new code
		// step1/3, load all of hive data to map, write to local file
		// if exits local file, load to map
		loadAllHiveData(start, start.plusDays(numFeatures));
		loadAllHiveData(start.plusDays(TARGET_AFTTER_DAYS), start.plusDays(TARGET_AFTTER_DAYS + numFeatures - 1));

		loadAllHiveData(evalStart, evalStart.plusDays(numFeatures));
		loadAllHiveData(evalStart.plusDays(TARGET_AFTTER_DAYS), evalStart.plusDays(TARGET_AFTTER_DAYS + numFeatures - 1));

		// step2/3, train data
		Map<String, Vector> samples = mapTransfer(start);
		out.println("train");
		Map<Integer, List<Vector>> samplesClass = train(start, samples);
		out.println("eval train samples");
		eval(evalStart, samples, samplesClass);

		// step3/3, eval data
		out.println("eval");
		Map<String, Vector> evalSamples = mapTransfer(evalStart);
		eval(evalStart, evalSamples, samplesClass);

	}

	private static Map<String, Vector> mapTransfer(LocalDate ld) {

		Map<String, Vector> accountIndex = new HashMap<>();
		Vector input = new RandomAccessSparseVector(numFeatures);

		Map<String, Map<String, Integer>> accountMaps = ldAccountMaps.get(ld);
		for (String accountId : accountMaps.keySet()) {
			int onlineDur = accountMaps.get(accountId).getOrDefault("online_dur", 0);
			input.setQuick(0, (double) onlineDur);
			for (int i = 1; i < numFeatures; i++) {
				int onlineDur2 = ldAccountMaps.get(ld.plusDays(i))
						.getOrDefault(accountId, new HashMap<>(0))
						.getOrDefault("online_dur", 0);
				input.setQuick(i, (double) onlineDur2);
			}

			accountIndex.put(accountId, input);
		}

		return accountIndex;
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
	
	private static double vectorSimilar(Vector eval, Vector sample){
		return eval.minus(sample).norm(2)/sample.norm(2);
	}

	private static void eval(LocalDate evalStart, Map<String, Vector> evalSamples, Map<Integer, List<Vector>> samplesClass) {
		out.println("eval start: ");
		Map<String, Integer> accountTargetValue = getTargetValue(evalStart, evalSamples);
		
		Integer[][] res = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };// abcd;

		for (Map.Entry<String, Vector> eval : evalSamples.entrySet()) {
			String accountId = eval.getKey();
			Vector input = eval.getValue();
			if (input == null)
				continue;
			
			double[] similar = {0.0, 0.0, 0.0};//绝对相似度
			int[] sum = {0, 0, 0};
			for(int i=0;i<=2;i++){
				for(Vector v:samplesClass.get(i)){
					sum[i] ++;
					similar[i] += vectorSimilar(input, v);
				}
			}
			
			double[] sr = {0.0, 0.0, 0.0};//平均相似度
			for(int i=0;i<=2;i++)
				sr[i] = similar[i]/sum[i];
			
			int targetValue = accountTargetValue.get(accountId);
			int predictValue = 1;
			if(sr[0]>sr[1] && sr[0]>sr[2])
				predictValue = 0;
			else if(sr[2]>sr[0] && sr[2]>sr[1])
				predictValue = 2;

			res[targetValue][predictValue]++;
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

	private static Map<Integer, List<Vector>> train(LocalDate ld, Map<String, Vector> samples) {
		out.println("train: " + ld.toString());
		Map<Integer, List<Vector>> sampleClass = new HashMap<>();
		for(int i=0;i<3;i++)
			sampleClass.put(i, new LinkedList<>());

		int[] sampleStat = { 0, 0, 0 };

		Map<String, Integer> accountTargetValue = getTargetValue(ld, samples);
		for (Map.Entry<String, Vector> sample : samples.entrySet()) {

			String accountId = sample.getKey();
			Vector input = sample.getValue();
			if (input == null)
				continue;

			int targetValue = accountTargetValue.get(accountId);
			if (targetValue < 0 || targetValue > 2)
				continue;

			sampleClass.get(targetValue).add(input);
			sampleStat[targetValue]++;

		}

		out.printf("train finish, lost cnt:%d, may cnt:%d, retain cnt: %d %n", sampleStat[0], sampleStat[1], sampleStat[2]);

		return sampleClass;

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
		int uptodate = Utils.dateDiff(ld, firstLoginDate);
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


	private static Map<String, Integer> getTargetValue(LocalDate ld, Map<String, Vector> samples) {
		
		Map<String, Vector> target = mapTransfer(ld.plusDays(TARGET_AFTTER_DAYS));
		
		Map<String, Integer> res = new HashMap<>(samples.size());
		
		int[] rowstat = {0, 0};
		Map<String, Map<String, Integer>> accountMaps = ldAccountMaps.get(ld.plusDays(TARGET_AFTTER_DAYS));
		for(String accountId:samples.keySet()){
			Map<String, Integer> cols = accountMaps.getOrDefault(accountId, new HashMap<>(0));
			
			try{
				Vector v = target.get(accountId);
	
				int nextHalfLoginDaycnt = 0;
				for(int i=0;i<7;i++){
					if(v.get(i) >= 1.0){
						nextHalfLoginDaycnt = 1;
						break;
					}	
				}
				
				int lastHalfLoginDaycnt = 0;
				for(int i=7;i<14;i++){
					if(v.get(i) >= 1.0){
						lastHalfLoginDaycnt = 1;
						break;
					}	
				}
				
				int targetValue = 1;// 将流失用户
				if (nextHalfLoginDaycnt > 0 && lastHalfLoginDaycnt > 0) {// 留存用户
					targetValue = 2;
				} else if (nextHalfLoginDaycnt == 0) {// 流失用户
					targetValue = 0;
				}
				
				res.put(accountId,  targetValue);
			} catch (Exception e){
				rowstat[1] ++;
			}
		}
		
		out.printf("get target null cnt: %d, get field exception cnt: %d %n", rowstat[0], rowstat[1]);
		return res;
	}

}
