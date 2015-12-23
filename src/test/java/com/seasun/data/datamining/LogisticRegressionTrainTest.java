package com.seasun.data.datamining;

import static org.junit.Assert.*;

import org.junit.Test;

public class LogisticRegressionTrainTest {

	@Test
	public void loadConfigFileTest() {
		LogisticRegressionTrain.loadConfigFile("./config.properties");
	}

}
