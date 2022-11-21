# Hadoop

```sh
hadoop distcp -su cm_dc_online,cmdaondfa47 -du xxx,xxx hdfs://yq01-wutai-hdfs.dmop.baidu.com:54310/app/ecom/cm/online_dc/online/new_orc/fc_lu/100p/20171106 hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310/app/ecom/fcr-model/lianglei02/dc
hadoop  distcp -D mapred.output.compress=false  -i -su "xxx,xxx" -du "dddd,dddd" -overwrite 源路径 目标路径

/home/nfs3/wangjingwen03/tools/hadooptools2/hadoop-client-yinglong/hadoop/bin/hadoop distcp afs://yinglong.afs.baidu.com:9902/user/fmflow/yunfan/job_output/task_4336/  afs://yinglong.afs.baidu.com:9902/user/rmp-mixer/rmp-individual/wangjingwen03/shuaku/ocr/top1_batch_7/
```



## hadoop 支持reduce多路输出的功能

- 一个reduce可以输出到多个part-xxxxx-X文件中，其中X是A-Z的字母之一，程序在输出<key,value>对的时候，在value的后面追加"#X"后缀，比如#A，输出的文件就是part-00000-A，不同的后缀可以把key,value输出到不同的文件中，方便做输出类型分类， #X仅仅用做指定输出文件后缀， 不会体现到输出的内容中。

- 在启动时，指定SuffixMultipleTextOutputFormat或者 SuffixMultipleSequenceFileOutputFormat

  ```shell
  # 前者对应于文本输入，后者于二进制输入
  -outputformat org.apache.hadoop.mapred.lib.SuffixMultipleTextOutputFormat
  -outputformat org.apache.hadoop.mapred.lib.SuffixMultipleSequenceFileOutputFormat
  
  ```

  



## Hadoop多列排序

```sh
bin/hadoop streaming -input /tmp/comp-test.txt -output /tmp/xx -mapper cat -reducer cat \
-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
-jobconf stream.num.map.output.key.fields=2 \
-jobconf num.key.fields.for.partition=1 \
-jobconf stream.map.output.field.separator=. \
-jobconf map.output.key.field.separator=. \
-jobconf mapred.reduce.tasks=5
```



- https://www.cnblogs.com/van19/p/5756448.html



## Hadoop 获取文件名

```python
filename = os.getenv("map_input_file")
```



