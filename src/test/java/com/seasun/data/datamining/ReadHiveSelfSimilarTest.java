package com.seasun.data.datamining;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

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
		
//		Vector input3 = new RandomAccessSparseVector(3);
//		input3.setQuick(0, 0);
//		input3.setQuick(1, 0);
//		input3.setQuick(2, 2);
//		System.out.println(input3.norm(2));
		
//		Vector v = new SequentialAccessSparseVector(2);
//		System.out.println(v.get(1));
//		v.setQuick(1, 0);
//		System.out.println(v.get(1));
		
		Map<Integer, Integer> m = new HashMap<>();
		m.put(1, 1);
		m.put(2,2);
		System.out.println(m);
//		for(Map.Entry<Integer, Integer> e:m.entrySet()){
//			m.remove(e.getKey());
//		}
		
//		m.forEach(
//				new BiConsumer<Integer, Integer>(){
//
//					@Override
//					public void accept(Integer t, Integer u) {
//						// TODO Auto-generated method stub
//						m.remove(t);
//					}}
//				);
		Iterator<Map.Entry<Integer, Integer>> it = m.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry<Integer, Integer> e = it.next();
			int k = e.getKey();
			it.remove();
		}
		
		System.out.println(m);
		
	}

}
