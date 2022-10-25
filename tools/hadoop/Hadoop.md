# Hadoop

```sh
hadoop distcp -su cm_dc_online,cmdaondfa47 -du xxx,xxx hdfs://yq01-wutai-hdfs.dmop.baidu.com:54310/app/ecom/cm/online_dc/online/new_orc/fc_lu/100p/20171106 hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310/app/ecom/fcr-model/lianglei02/dc
hadoop  distcp -D mapred.output.compress=false  -i -su "xxx,xxx" -du "dddd,dddd" -overwrite 源路径 目标路径

/home/nfs3/wangjingwen03/tools/hadooptools2/hadoop-client-yinglong/hadoop/bin/hadoop distcp afs://yinglong.afs.baidu.com:9902/user/fmflow/yunfan/job_output/task_4336/  afs://yinglong.afs.baidu.com:9902/user/rmp-mixer/rmp-individual/wangjingwen03/shuaku/ocr/top1_batch_7/
```



## hadoop 支持reduce多路输出的功能

- 一个reduce可以输出到多个part-xxxxx-X文件中，其中X是A-Z的字母之一，程序在输出<key,value>对的时候，在value的后面追加"#X"后缀，比如#A，输出的文件就是part-00000-A，不同的后缀可以把key,value输出到不同的文件中，方便做输出类型分类， #X仅仅用做指定输出文件后缀， 不会体现到输出的内容中。

- 在启动时，指定SuffixMultipleTextOutputFormat或者 SuffixMultipleSequenceFileOutputFormat


