package com.seasun.data.datamining;

import static org.junit.Assert.*;

import org.junit.Test;

public class UtilsTest {

	@Test
	public void getOrDefaultTest() {
		double d = Utils.getOrDefault("classify_value", 0.2);
		System.out.println(d);
		
		int i = Utils.getOrDefault("train_passes", 100);
		System.out.println(i);
		
		String s = Utils.getOrDefault("sample_dir", "abc");
		System.out.println(s);

	}
	
	@Test
	public void printResMatrixTest(){
		Integer[][] res = { {1,2,3}, {4,4,4}, {5,5,5} };
		Utils.printResMatrix(res);
	}

}
