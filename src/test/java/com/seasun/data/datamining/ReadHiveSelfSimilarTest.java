package com.seasun.data.datamining;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class ReadHiveSelfSimilarTest {

	@Test
	public void test() {
		Vector input = new RandomAccessSparseVector(3);
		System.out.println(input.toString() + ":" + input.norm(2));
		
		input.set(0, 0);
		input.set(1, 1);
		input.set(2, 0);
		System.out.println(input.toString() + ":" + input.norm(2));
		System.out.println(input.getNumNonZeroElements());
		
//		System.out.println(input.minus(input).norm(2)/input.norm(2));
//		
//		Vector input2 = new RandomAccessSparseVector(3);
//		input2.set(0, 1);
//		System.out.println(input2.minus(input).norm(2)/input.norm(2));
//		System.out.println(input.minus(input).norm(2)/input2.norm(2));
		
//		Vector input3 = new RandomAccessSparseVector(3);
//		input3.setQuick(0, 0);
//		input3.setQuick(1, 0);
//		input3.setQuick(2, 2);
//		System.out.println(input3.norm(2));
		
//		Vector v = new SequentialAccessSparseVector(2);
//		System.out.println(v.get(1));
//		v.setQuick(1, 0);
//		System.out.println(v.get(1));
		

		
	}
	
	@Test
	public void mapTransferTest(){
		Map<LocalDate, Map<String, Map<String, Integer>>> ldAccountMaps = ReadHiveSelfSimilar.ldAccountMaps;
		Map<String, Integer> cols = new HashMap<>();
		cols.put("online_dur", 10);
		cols.put("role_level", 20);
		
		int numFeatures = 14;

		
		LocalDate start = LocalDate.parse("2016-03-01");
		LocalDate end = LocalDate.parse("2016-03-14");
		for (LocalDate ld = start; !ld.isAfter(end.plusDays(numFeatures-1)); ld = ld.plusDays(1)) {
			Map<String, Map<String, Integer>> map = new HashMap<>();
			map.put("account_1", new HashMap<String, Integer>(cols));
			map.put("account_2", new HashMap<String, Integer>(cols));
			ldAccountMaps.put(ld, map);
		}
		
		System.out.println( ReadHiveSelfSimilar.mapTransfer(start, end, numFeatures, true) );
		
	}

}
