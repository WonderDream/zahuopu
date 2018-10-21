from __future__ import division
import pandas as pd
import os
import os.path
from sqlalchemy import types, create_engine
import math
import cx_Oracle


engine = create_engine('oracle+cx_oracle://user:pass@localhost:1521/?service_name=database', encoding="utf-8")
conn = cx_Oracle.connect('user/pass@localhost/database')

dataDir = 'E:/Data/DataSet'
prefix = 'dataset'.upper()
for parent, dirname, filenames in os.walk(dataDir):
    for filename in filenames:
        fullpath = os.path.join(parent,filename)
        tableName = prefix + filename.split('.')[0].upper()
        if not engine.has_table(tableName):
            print('save data for ' + tableName + '...\n')
            
            df = pd.read_csv(fullpath, sep = '\t', header = 0, encoding= 'utf-8', nrows = 10)
            
            converters = {k:unicode for k in range(len(df.columns))}
            df = pd.read_csv(fullpath, sep = '\t', header = 0, encoding= 'utf-8', converters = converters)
            dtyp = {c:types.VARCHAR(max(df[c].str.len().max(), 1)) for c in df.columns[df.dtypes == 'object'].tolist()}
            
            #create the table using to_sql
            df.iloc[:10].to_sql(tableName, engine, dtype = dtyp, if_exists='replace', index = False)


            cursor = conn.cursor()
            cursor.execute('truncate table ' + tableName)
            conn.commit()

            sql = 'insert into ' + tableName + '('
            valueStubs = ''
            for k in range(len(df.columns)):
                col = df.columns[k]
                sql = sql + col
                valueStubs = valueStubs + ':' + str(k + 1)
                if k != len(df.columns) - 1:
                    sql = sql + ','
                    valueStubs = valueStubs + ','
            
            sql = sql + ') values (' + valueStubs + ')'

            cursor.prepare(sql)
            
            batchSize = 1000
            batchCnt = int(math.ceil(len(df)/batchSize))
                          
            for k in range(batchCnt):
                batchBegin = k*batchSize
                batchEnd = min((k + 1)*batchSize, len(df))
                batch = df.iloc[batchBegin:batchEnd]
                if len(batch) > 0:
                    cursor.executemany(None, batch.values.tolist())
                    conn.commit()
conn.commit()
conn.close()

        



        


