# python连接数据库

## 1、orcale

```python
# 待验证
import os
import cx_Oracle
db_configs= {
        'drivername':'oracle',
        'host':'*.*.*.*',
        'port':1521,
        'username': 'username',
        'password': 'password',
        'database': 'orcl'
}

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
dsn = cx_Oracle.makedsn(db_configs['host'], db_configs['port'], db_configs['database'])
conn = cx_Oracle.connect(db_configs['username'], db_configs['password'], dsn)
oracle_from = conn.cursor()

sql_statement = ''
df = pd.DataFrame(oracle_from.execute(sql_statement).fetchall(), columns=['',''])

```



## 2、PostgreSQL

```python
# 在python中执行SQL语句
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
import pandas as pd

configs = {
        'drivername':'postgres',
        'host':'*.*.*.*',
        'port':'5432',
        'username':'username',
        'password':'password',
        'database':'DB'
    }
url = URL(**configs)
engine = create_engine(url)
connection = engine.connect()

my_query = 'SELECT * FROM t_telecom_cdr_ex Limit 10'
results = connection.execute(my_query).fetchall()
df = pd.DataFrame(results)
print(df)
```



```python
# 用python的API进行数据操作
# 参考资料：https://www.pythonsheets.com/notes/python-sqlalchemy.htmlc
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine,MetaData,select
import pandas as pd

configs = {
        'drivername':'postgres',
        'host':'*.*.*.*',
        'port':'5432',
        'username':'username',
        'password':'password',
        'database':'DB'
    }
url = URL(**configs)
engine = create_engine(url)
connection = engine.connect()

metadata = MetaData(engine,reflect=True)
t_telecom_cdr_ex = metadata.tables['t_telecom_cdr_ex']

select_st = select([t_telecom_cdr_ex]).where(t_telecom_cdr_ex.columns.charge_phone_no=='13426286962')
res = connection.execute(select_st)

cols = [i.name for i in t_telecom_cdr_ex.columns]   # 获取各列的列名

for _row in res:
    print(_row)
```

