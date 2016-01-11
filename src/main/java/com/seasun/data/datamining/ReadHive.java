package com.seasun.data.datamining;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import com.github.dataswitch.util.HadoopConfUtil;

public class ReadHive {

	public static void main(String[] args) {

        //String uri = "hdfs://xxxx:8020" + "/user/hive/warehouse/lib/col_iplib.txt";
//        String uri = "hdfs://namenodem:50070/hive/warehouse/fig.db/lost_user/dt=2015-12-30/000000_0";
//        String uri = "hdfs://namenodem:50070/hive/warehouse/fig.db/lost_user/dt=2015-12-30/000000_0";
		String path = "hdfs:/hive/warehouse/fig.db/lost_user/dt=2015-12-30/000000_0";
        FileSystem fs = null;
        FSDataInputStream in = null;
        BufferedReader br = null;
        try {
            fs = HadoopConfUtil.getFileSystem(null, null);
            in = fs.open(new Path(path));
            br = new BufferedReader(new InputStreamReader(in));
            String readline;
            int counter = 0;
            while ((readline = br.readLine()) != null && counter <=10) {
                String[] pars = readline.split("\001");
                System.out.println(pars[1]);
                counter++;
            }	
        } catch (IOException e) {
        	e.printStackTrace();
            System.out.println(e.getMessage());
        } finally {
            IOUtils.closeStream(in);
        }
	}

}
