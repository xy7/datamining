package com.seasun.data.datamining;

import static org.junit.Assert.*;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class ReadHiveSelfSimilarTest {

	@Test
	public void test() {
//		Vector input = new RandomAccessSparseVector(3);
//		System.out.println(input.toString() + ":" + input.norm(2));
//		
//		input.set(0, 0);
//		input.set(2, 0);
//		System.out.println(input.toString() + ":" + input.norm(2));
//		System.out.println(input.nonZeroes().iterator().hasNext());
//		
//		System.out.println(input.minus(input).norm(2)/input.norm(2));
//		
//		Vector input2 = new RandomAccessSparseVector(3);
//		input2.set(0, 1);
//		System.out.println(input2.minus(input).norm(2)/input.norm(2));
//		System.out.println(input.minus(input).norm(2)/input2.norm(2));
		
		Vector input3 = new RandomAccessSparseVector(3);
		input3.setQuick(0, 0);
		input3.setQuick(1, 0);
		input3.setQuick(2, 2);
		System.out.println(input3.norm(2));
	}

}
