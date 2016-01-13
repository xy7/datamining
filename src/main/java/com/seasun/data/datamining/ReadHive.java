package com.seasun.data.datamining;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
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
	private static String table = "/hive/warehouse/fig.db/lost_user/dt=";
	
	private static PrintWriter output = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
	
	private static Map<String, Integer> lost = new HashMap<>();
	private static OnlineLogisticRegression lr;
	
	static {
		try {
			hdfs = HadoopConfUtil.getFileSystem(null, null);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		lr = new OnlineLogisticRegression(2, LogisticRegressionTrain.numFeatures, new L1());
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
		        Vector input = new RandomAccessSparseVector(LogisticRegressionTrain.numFeatures);
		        try {
		        	accountId = parseLine(line, input);
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
					return false;
				}
		        return true;
			}
		});
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
					String[] cols = line.split(LogisticRegressionTrain.COLUMN_SPLIT);
			    	String accountId = cols[1];
					
					String appId = cols[0];
					if(!appId.equals(APPID))
						return true;
					
					int last7LoginDaycnt = Integer.parseInt(cols[34] );
					int targetValue = last7LoginDaycnt>0?1:0;
					lost.put(accountId, targetValue);
					return true;
				} catch(Exception e) {
					e.printStackTrace();
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
			}// for file
			
			output.printf(Locale.ENGLISH, "pass %d: all(%d) sucess(%d) %n"
					, pass, all, suc);
		}// for pass
	}
	
	//返回帐号id
	public static String parseLine(String line, Vector featureVector) throws Exception {
		featureVector.setQuick(0, 1.0);//填充常量 k0
		int[] index = LogisticRegressionTrain.index;
		List<String> values = Arrays.asList(line.split(LogisticRegressionTrain.COLUMN_SPLIT));
		if(values.size() < index[index.length-1] + 1)
			throw new Exception("parse error, columns size to small: " + values.size());
		
		for (int i = 0; i < index.length; i++) {
			String s = values.get(index[i]);
			if(s.equals("\\N"))//null值替换为空值
				s = "0";
			featureVector.setQuick(i + 1, Double.parseDouble(s));
		}
		
		String accountId = values.get(1);
		
		String appId = values.get(0);
		if(!appId.equals(APPID))
			return null;
		
		return accountId;
	}


}
