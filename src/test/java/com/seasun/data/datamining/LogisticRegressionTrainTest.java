package com.seasun.data.datamining;

import static org.junit.Assert.*;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

import org.junit.Test;

public class LogisticRegressionTrainTest {

	@Test
	public void loadConfigFileTest() {
		LogisticRegressionTrain.loadConfigFile("./config.properties");
	}
	
	@Test
	public void genSql(){
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
		LocalDate ld = LocalDate.parse("2015-11-01");
		
		String sql = "INSERT OVERWRITE LOCAL DIRECTORY '/tmp/fig_app_user/"+ ld.format(formatter) +"/'\n"
				+ "select a.*, datediff(dt, first_login_date),(case when b.account_id is null then 1 else 0 end)\n"
				+ "from(\n"
				+ "select *\n"
				+ "from fig.fig_app_user_v\n"
				+ "where dt = '"+ ld.format(formatter) +"'\n"
				+ "and user_type = 'account'\n"
				+ "and app_id = '1024appid'\n"
				+ "and first_login_date <= '"+ ld.minusDays(13).format(formatter) +"' \n"
				+ "and datediff(last_login_date, first_login_date) >= 7\n"
				+ "and last7_login_daycnt >= 1\n"
				+ "and (last2_login_daycnt = 0\n"
				+ " or last3_login_daycnt - last1_login_daycnt = 0\n"
				+ " or last4_login_daycnt - last2_login_daycnt = 0\n"
				+ " or last5_login_daycnt - last3_login_daycnt = 0\n"
				+ " or last6_login_daycnt - last4_login_daycnt = 0\n"
				+ " or last7_login_daycnt - last5_login_daycnt = 0\n"
				+ " )\n"
				+ ") a left join (\n"
				+ "select account_id\n"
				+ "from fig.fig_app_user_v\n"
				+ "where dt = '"+ ld.plusDays(23).format(formatter) +"'\n" //16
				+ "and user_type = 'account'\n"
				+ "and app_id = '1024appid'\n"
				+ "and last14_login_daycnt > 0\n"
				+ ") b\n"
				+ "on a.account_id = b.account_id\n"
				+ ";"
				;
		System.out.println(sql);
	}

}
