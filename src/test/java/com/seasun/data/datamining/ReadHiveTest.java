package com.seasun.data.datamining;

import static org.junit.Assert.*;

import java.time.LocalDate;

import org.junit.Test;

public class ReadHiveTest {

	@Test
	public void test() {
		String s = "\1";
		System.out.println(s);
	}
	
	@Test
	public void dayDiffTest(){
		LocalDate start = LocalDate.parse("2015-08-01");
		LocalDate end   = LocalDate.parse("2015-09-03");
		System.out.println(ReadHive.dayDiff(start, end) );
	}

}
