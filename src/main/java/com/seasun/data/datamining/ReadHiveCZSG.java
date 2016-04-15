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

	public static void main(String[] args) throws Exception {

		Utils.loadConfigFile("./config.properties.hive");
		APPID = Utils.getOrDefault("appid", APPID);
		SCORE_FREQ = Utils.getOrDefault("score_freq", 0);
		MAY_LOST_REPEAT_CNT = Utils.getOrDefault("may_lost_repeat_cnt", 1);
		RETAIN_THRESHOLD = Utils.getOrDefault("retain_threshold", 15);

		String serialFileName = Utils.getOrDefault("serial_file_name", "./czsg_serial_data.out");

		File file = new File(serialFileName);
		if (file.exists()) {
			out.println("serial file exists read to Map");
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
			ldSamples = (Map<LocalDate, List<Sample>>) in.readObject();
			in.close();
		}

		LocalDate start = LocalDate.parse(Utils.getOrDefault("train_start", "2015-11-01"));
		LocalDate end = LocalDate.parse(Utils.getOrDefault("train_end", "2015-11-01"));
		int trainPass = Utils.getOrDefault("train_pass", 1);
		for (int i = 0; i < trainPass; i++) {
			out.printf(Locale.ENGLISH, "--------pass: %2d ---------%n", i);
			for (LocalDate ld = start; !ld.isAfter(end); ld = ld.plusDays(1)) {
				train(ld);
			}
		}

		Utils.saveModel(lr);

		LocalDate evalStart = LocalDate.parse(Utils.getOrDefault("eval_start", "2015-11-01"));
		LocalDate evalEnd = LocalDate.parse(Utils.getOrDefault("eval_end", "2015-11-01"));

		for (LocalDate ld = evalStart; !ld.isAfter(evalEnd); ld = ld.plusDays(1)) {
			eval(ld);
		}

		if (!file.exists()) {
			out.println("serial file not exists, now write to it");
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file));
			out.writeObject(ldSamples);
			out.close();
		}

	}

	static class Sample implements Serializable {
		private static final long serialVersionUID = -5983545824659336774L;
		public int targetValue;
		public transient Vector input;

		public Sample(int targetValue, Vector input) {
			this.targetValue = targetValue;
			this.input = input;
		}

		private void writeObject(ObjectOutputStream out) throws IOException {
			out.defaultWriteObject();
			for (int i = 0; i < numFeatures; i++)
				out.writeDouble(input.get(i));
		}

		private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
			in.defaultReadObject();
			input = new RandomAccessSparseVector(numFeatures);
			for (int i = 0; i < numFeatures; i++)
				input.setQuick(i, in.readDouble());
		}

		@Override
		public String toString() {
			return "Sample [targetValue=" + targetValue + ", input=" + input + "]";
		}

	}

	private static Map<LocalDate, List<Sample>> ldSamples = new HashMap<>();

	private static void eval(LocalDate ld) {
		out.println("eval: " + ld.toString());
		Integer[][] res = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };// abcd;

		if (ldSamples.containsKey(ld)) {
			List<Sample> samples = ldSamples.get(ld);
			for (Sample sample : samples) {
				evalSample(sample, res);
			}
		} else {
			List<Sample> samples = new LinkedList<>();
			ldSamples.put(ld, samples);

			getTargetValue(ld);

			RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld);
			if (lfss == null)
				return;

			Map<String, Integer> lostLd = lost.get(ld);
			Utils.analysisHdfsFiles(lfss, new LineHandler() {
				@Override
				public boolean handle(String line) {
					String accountId;
					Vector input = new RandomAccessSparseVector(numFeatures);
					try {
						accountId = parseLine(line, input, ld);
						if (accountId == null)
							return false;
						int targetValue = lostLd.get(accountId);

						Sample sample = new Sample(targetValue, input);
						samples.add(sample);

						evalSample(sample, res);

					} catch (Exception e) {
						// e.printStackTrace();
						// out.println(e);
						return false;
					}
					return true;
				}
			});

		} // else

		int all = 0;
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				all += res[i][j];
		
		out.println("result matrix:");
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				out.printf("%2.4f \t", (double) res[i][j] / all);
			}
			out.printf("%n");
		}
		
	}

	private static void evalSample(Sample sample, Integer[][] res) {
		Vector input = sample.input;
		int targetValue = sample.targetValue;

		Vector score = lr.classify(input);

		double s1 = score.get(0);
		double s2 = score.get(1);
		double s0 = 1 - s1 - s2;

		int predictValue = 0;
		if (s1 >= s2 && s1 >= s0)
			predictValue = 1;
		if (s2 > s1 && s2 > s0)
			predictValue = 2;
		
		res[targetValue][predictValue] ++;

	}

	private static void train(LocalDate ld) {
		out.println("train: " + ld.toString());
		sampleCnt = 0;
		if (ldSamples.containsKey(ld)) {
			List<Sample> samples = ldSamples.get(ld);
			for (Sample sample : samples)
				trainSample(sample);

			eval(ld);
			return;
		}

		List<Sample> samples = new LinkedList<>();
		ldSamples.put(ld, samples);

		getTargetValue(ld);
		RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld);
		if (lfss == null)
			return;

		Map<String, Integer> lostLd = lost.get(ld);
		int[] res = new int[3];// parse error, get target error, sucess
		rowstat = new int[] { 0, 0, 0, 0, 0 };
		Utils.analysisHdfsFiles(lfss, new LineHandler() {
			@Override
			public boolean handle(String line) {
				Vector input = new RandomAccessSparseVector(numFeatures);
				String accountId = parseLine(line, input, ld);
				if (accountId == null) {
					res[0]++;
					return false;
				}

				int targetValue = lostLd.getOrDefault(accountId, -1);
				if (targetValue == -1) {
					res[1]++;
					return false;
				}

				Sample sample = new Sample(targetValue, input);
				samples.add(sample);

				trainSample(sample);

				res[2]++;

				return true;
			}
		});

		out.printf(
				"columns size too small or appid not %s: %d, first_login_date error: %d, uptodate < 14: %d, last14LoginDaycnt not in[2,13]: %d  %n"
				, APPID, rowstat[0], rowstat[1], rowstat[2], rowstat[3]);
		out.printf("parse error: %d, get target error: %d, sucess: %d %n", res[0], res[1], res[2]);

		eval(ld);
	}

	private static int sampleCnt = 0;
	private static int MAY_LOST_REPEAT_CNT;

	private static void trainSample(Sample sample) {
		int targetValue = sample.targetValue;
		Vector input = sample.input;
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

	private static Map<String, String> lineSplit(String line) {

		String[] cols = line.split("\1");
		String appId = cols[0];
		if (!appId.equals(APPID))
			return null;
		Map<String, String> res = new HashMap<>();

		res.put("app_id", cols[0]);
		res.put("account_id", cols[1]);
		res.put("device_id", cols[2]);
		res.put("mac", cols[3]);

		String mapStr = cols[4];
		String mapInt = cols[5];

		res.putAll(parse2map(mapStr));
		res.putAll(parse2map(mapInt));

		return res;
	}

	private static Map<String, String> parse2map(String mapInt) {
		String[] intCols = mapInt.split("\2");
		Map<String, String> res = new HashMap<>();
		for (String col : intCols) {
			// out.println("167: " + col);
			String[] kv = col.split("\3");
			String value = "";
			if (kv.length >= 2)
				value = kv[1];

			res.put(kv[0], value);
		}
		return res;
	}

	private static void getTargetValue(LocalDate ld) {
		out.println("getTargetValue: " + ld.toString());
		if (lost.containsKey(ld))
			return;
		RemoteIterator<LocatedFileStatus> lfss = Utils.listHdfsFiles(ld.plusDays(30));
		if (lfss == null)
			return;

		Map<String, Integer> lostLd = new HashMap<>();
		Utils.analysisHdfsFiles(lfss, new LineHandler() {
			@Override
			public boolean handle(String line) {
				try {
					Map<String, String> cols = lineSplit(line);
					if (cols == null)
						return true;
					// out.println("line column size: " + cols.size() +
					// "  values: " + cols);
					String accountId = cols.get("account_id");

					int lastAllLoginDaycnt = Integer.parseInt(cols.get("last30_login_daycnt"));
					int lastHalfLoginDaycnt = Integer.parseInt(cols.get("last14_login_daycnt"));
					int nextHalfLoginDaycnt = lastAllLoginDaycnt - lastHalfLoginDaycnt;
					//int last30LoginDaycnt = Integer.parseInt(cols.get("last30_login_daycnt"));

					int targetValue = 1;// 将流失用户
					if (nextHalfLoginDaycnt > 0 && lastHalfLoginDaycnt > 0) {// 留存用户
						targetValue = 2;
					} else if (nextHalfLoginDaycnt == 0 ) {// 流失用户
						targetValue = 0;
					}

					lostLd.put(accountId, targetValue);
					return true;
				} catch (NullPointerException e) {
					// e.printStackTrace();
					// out.println(e);
					return false;
				}
			}

		});

		lost.put(ld, lostLd);
	}

	// 返回帐号id
	private static int[] rowstat;

	private static String parseLine(String line, Vector featureVector, LocalDate ld) {

		featureVector.setQuick(0, 1.0);// 填充常量 k0

		Map<String, String> cols = lineSplit(line);

		if (cols == null || cols.size() < 60) {
			// out.println("parse error, columns size to small: " +
			// cols.size());
			rowstat[0]++;
			return null;
		}

		int i = 1;
		for (String k : trainIndex) {
			String s = cols.get(k);
			if (s.equals("\\N"))// null值替换为0
				s = "0";
			featureVector.setQuick(i, Double.parseDouble(s));
			i++;
		}

		String fistLoginDate = cols.get("first_login_date");
		if (fistLoginDate == null || fistLoginDate.length() != 10) {
			// out.println("fistLoginDate format error: " + fistLoginDate);
			rowstat[1]++;
			return null;
		}
		LocalDate firstLoginDate = LocalDate.parse(fistLoginDate);
		int uptodate = Utils.dateDiff(ld, firstLoginDate);
		if (uptodate < 14) {
			// out.println("uptodate < 14: " + firstLoginDate);
			rowstat[2]++;
			return null;
		}

		int last14LoginDaycnt = Integer.parseInt(cols.get("last14_login_daycnt"));
		if (last14LoginDaycnt < 2 || last14LoginDaycnt > 13) {
			// out.println("last7LoginDaycnt < 1");
			rowstat[3]++;
			return null;
		}

		featureVector.setQuick(i, uptodate);

		String accountId = cols.get("account_id");

		return accountId;
	}

}
