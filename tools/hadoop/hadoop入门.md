# 什么是mr操作

MapReduce是一种编程模型和算法，用于处理大规模数据集。它的基本原理是将大数据集分解成许多小的数据块，然后在多个计算节点上并行处理这些数据块。

Map操作是MapReduce的第一阶段，它将输入的数据集分成较小的子集，并分配给各个计算节点进行并行处理。每个计算节点只处理本地存储的数据，这样就可以大大降低数据传输的开销。在Map阶段，输入数据被处理成键值对（key-value pairs），这些键值对根据键的排序进行分组，然后传递给Reduce阶段。

Reduce操作是MapReduce的第二阶段，它对Map阶段产生的键值对进行汇总和归约，生成最终的结果。Reduce阶段会接收Map阶段输出的键值对，根据键进行排序和分组，然后对每个键的值进行归约操作，生成最终的结果。

MapReduce操作之所以有效，主要有以下几个原因：

1. 并行处理：MapReduce允许数据并行处理，将大规模数据集分成小块，并在多个计算节点上同时处理这些数据块。这大大提高了数据处理的速度和效率。
2. 数据本地化：在Map阶段，数据被分配给各个计算节点进行处理，减少了数据传输的开销。这样可以提高数据处理的效率。
3. 自动容错：MapReduce会自动处理节点故障和数据丢失等问题，确保数据的可靠性和完整性。
4. 通用性：MapReduce提供了一种通用的数据处理框架，可以处理各种类型的数据，并进行各种类型的分析和挖掘。

# 一个典型的mr操作

一个典型的MapReduce操作包括以下步骤：

1. **输入**：首先，需要输入数据。例如，输入可能是一个存储在HDFS（Hadoop分布式文件系统）上的大型文本文件。
2. **Map阶段**：在Map阶段，程序读取输入数据，并执行指定的映射函数。这个函数将输入数据转换为一系列的键值对（key-value pairs）。例如，如果输入数据是一个文本文件，映射函数可以逐行读取文件，将每一行转换为一个键值对，其中键是行的位置（行号），值是行的内容。
3. **Shuffle阶段**：在Shuffle阶段，程序根据键对所有的键值对进行排序和分组，以便相同的键都聚集在一起。这样，具有相同键的所有键值对都可以传递给同一个Reduce任务。
4. **Reduce阶段**：在Reduce阶段，程序接收来自Shuffle阶段的数据，并执行指定的归约函数。这个函数将处理聚集在一起的键值对，并生成最终的输出结果。例如，如果输入数据是一个包含多个网页的文本文件，归约函数可以计算每个URL的频率，并将结果输出到一个文件中。
5. **输出**：最后，输出结果是Reduce阶段的输出。例如，它可以是一个包含每个URL及其出现次数的文本文件。

这个过程的主要优点是它可以在大量计算节点上并行处理数据，从而大大提高了处理大规模数据集的速度和效率。

# mr操作细节

1. 一个mr操作会对数据进行kv排序，其中map函数处理输入，而reduce函数接受kv对，并且进行聚合
2. 平台透明地分割数据分给每个map任务的slot，我们不关心这个分发策略会如何影响效率。
3. 排序操作发生在reduce任务之前。







1. 文件路径操作：

- hdfs dfs -mkdir dir：创建文件夹
- hdfs dfs -rmr dir：删除文件夹
- hdfs dfs -ls：查看目录文件信息
- hdfs dfs -lsr：递归查看文件目录信息
- hdfs dfs -stat path：返回指定路径的信息

1. 空间大小查看：

- hdfs dfs -du -h dir：按照适合阅读的形式人性化显示文件大小
- hdfs dfs -dus uri：递归显示目标文件的大小
- hdfs dfs -du path/file：显示目标文件file的大小

1. 目录和文件操作：

- hadoop fs –ls [文件目录]：查看目录下的文件列表
- hadoop fs –put [本机目录] [hadoop目录]：将本机文件夹存储至Hadoop上
- hadoop fs –mkdir [目录]：在Hadoop指定目录内创建新目录
- hadoop fs -touchz /lance/tmp.txt：在Hadoop指定目录下新建一个文件，使用touchz命令
- hadoop fs –put [本机地址] [hadoop目录]：将本机文件存储至Hadoop上

Hadoop的DFS参数是指分布式文件系统（Distributed File System），它是Hadoop的核心组件之一。DFS参数用于配置Hadoop分布式文件系统的相关参数，以控制Hadoop集群的行为和性能。

以下是DFS参数的一些常见功能：

1. 指定HDFS的地址和端口：通过DFS参数可以配置HDFS的主节点（NameNode）地址和端口号，以及其他相关服务如Secondary NameNode和DataNode的地址和端口号。
2. 控制HDFS的副本因子：DFS参数可以配置HDFS的副本因子，即每个文件块在集群中的副本数量。副本因子可以在文件创建时指定，也可以在运行时动态调整。
3. 配置HDFS的存储容量和磁盘类型：DFS参数可以配置HDFS中各个DataNode的存储容量和磁盘类型，以优化存储效率和数据可靠性。
4. 调整HDFS的并发连接数：DFS参数可以限制同时连接到HDFS集群的用户和客户端的数量，以防止过度负载。
5. 控制HDFS的安全模式：DFS参数可以配置Hadoop集群的安全模式，以确保只有经过授权的用户才能访问HDFS中的数据。

总之，DFS参数提供了丰富的配置选项，用于优化Hadoop分布式文件系统的性能、可靠性和安全性。根据实际需求和集群环境，合理配置DFS参数可以提高Hadoop集群的运行效率和数据管理能力。

Hadoop的默认DFS（分布式文件系统）参数可以通过修改Hadoop的配置文件来设置。Hadoop的配置文件通常位于`etc/hadoop`目录下，其中最重要的配置文件是`hdfs-site.xml`。

要设置DFS的默认参数，你需要修改`hdfs-site.xml`文件中的相应属性。以下是一些常见的DFS参数及其默认值：

1. `dfs.namenode.name.dir`：存储元数据的目录。默认值是`${hadoop.tmp.dir}/dfs/name`。
2. `dfs.datanode.data.dir`：存储数据的目录。默认值是`${hadoop.tmp.dir}/dfs/data`。
3. `dfs.replication`：数据的副本因子。默认值是3。
4. `dfs.block.size`：HDFS中块的大小，以字节为单位。默认值是128MB。
5. `dfs.namenode.handler.count`：NameNode的处理器数量。默认值是10。
6. `dfs.datanode.handler.count`：DataNode的处理器数量。默认值是3。
7. `dfs.client.block.write.replace-datanode-on-failure.policy`：当客户端写入失败时，是否替换DataNode的策略。默认值是`NEVER`。

要修改这些参数，你可以直接编辑`hdfs-site.xml`文件，或者在Hadoop的配置目录下创建一个新的配置文件（例如`my-hdfs-site.xml`），并在启动Hadoop时通过`-conf`选项指定该配置文件。