package com.seasun.data.datamining;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

public class ReadHiveTest {

	@Test
	public void test() {
		Map<Integer, Integer> m = new HashMap<>();
		int a = m.getOrDefault(1, -1111);
		System.out.println(a);
	}
	
	@Test
	public void dayDiffTest(){
		LocalDate start = LocalDate.parse("2015-08-01");
		LocalDate end   = LocalDate.parse("2015-09-01");
		System.out.println(ReadHive.dateDiff(start, end) );
		System.out.println(ReadHive.dateDiff(end, start) );
		
	}

}
