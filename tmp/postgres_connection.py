# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:28:28 2016

@author: ewood
"""

#### import python libraries
import psycopg2
import numpy as np
import pandas as pd

#host_name = 'tsdc-server'
host_name = 'localhost'
port_name = '5432'
user_name = 'postgres'
password_name = 'Fleet_DNA#'

#### function definitions
def query_table(query,db_name):
    conn = psycopg2.connect(host = host_name, port = port_name, database = db_name, user = user_name, password = password_name)
    cur = conn.cursor()
    cur.execute(query)
    colnames = [desc[0] for desc in cur.description]
    st = dict()
    for cname in colnames:
        st[cname] = []
    for record in cur:
        for i in range(len(colnames)):
            st[colnames[i]].append( record[i] )
    for i in range(len(colnames)):
        st[colnames[i]] = np.array( st[colnames[i]] )
    cur.close()
    conn.close()
    return st

def query_table_df(query, conn_info):
    conn = psycopg2.connect(host = conn_info['host_name'], port = conn_info['port_name'], \
    database = conn_info['db_name'], user = conn_info['user_name'], password = conn_info['password_name'])

    cur = conn.cursor()
    cur.execute(query)
    colnames = [desc[0] for desc in cur.description]
    st = pd.DataFrame(cur.fetchall(), columns = colnames)
    cur.close()
    conn.close()
    return st

def oneway_query(quer,db_name):
    conn = psycopg2.connect(host = host_name, port = port_name, database = db_name, user = user_name, password = password_name)
    cur = conn.cursor()
    cur.execute(quer)
    cur.close()
    conn.commit()
    conn.close()

def drop_table(table_name,db_name):
    quer = """DROP TABLE IF EXISTS """+table_name+""";"""
    conn = psycopg2.connect(host = host_name, port = port_name, database = db_name, user = user_name, password = password_name)
    cur = conn.cursor()
    cur.execute(quer)
    conn.commit()
    cur.close()
    conn.close()

def create_v_pass_table(table_name,db_name):
    fields_str = """
        sampno integer,
        vehno integer,
        tripno integer,
        start_ind integer,
        end_ind integer,
        start_time_local timestamp without time zone,
        end_time_local timestamp without time zone,
        start_mph double precision,
        end_mph double precision,
        delta_elev_m double precision,
        prev_start_mph double precision,
        prev_end_mph double precision,
        prev_net_id numeric,
        next_start_mph double precision,
        next_end_mph double precision,
        next_net_id numeric,
        net_id numeric,
        seconds double precision,
        zero_seconds double precision,
        avg_accel_mphps double precision,
        stop_cnt integer,
        miles double precision,
        gallons double precision,
        mean_mph double precision,
        delta_mph double precision,
        mpg double precision
        """
    quer = """CREATE TABLE """+table_name+""" (
    """+fields_str+"""
    );"""
    conn = psycopg2.connect(host = host_name, port = port_name, database = db_name, user = user_name, password = password_name)
    cur = conn.cursor()
    cur.execute(quer)
    conn.commit()
    cur.close()


def append_table(table_name, db_name, data):
    fnl_upl = []
    keys = data.keys()
    for i in range(len(data[keys[0]])):
        load = ()
        for a in range(len(keys)):
#            if (type(data[keys[a]][i])==np.int64) | (type(data[keys[a]][i])==np.bool_):
            if (type(data[keys[a]][i])==np.int64):
                load += (int(data[keys[a]][i]),)
            elif (keys[a]=='grdsrc1'):
                load += (str(int(data[keys[a]][i])),)
            else:
                load += (data[keys[a]][i],)
        upl = dict(zip(keys,load))
        fnl_upl.append(upl)
    #    insrt_quer = """INSERT INTO adas.adarg_hh(hh_id,road_id,dist_mi,avg_grade,maxmin_grade)
    #        VALUES ( %(hh_id)s, %(road_id)s, %(dist_mi)s, %(avg_grade)s, %(maxmin_grade)s );"""
    insrt_quer1 = """INSERT INTO """+table_name+"""("""
    insrt_quer2 = """) VALUES ( """
    insrt_quer3 = """);"""
    for a in range(len(keys)):
        if a==0:
            insrt_quer1 = insrt_quer1 + keys[a]
            insrt_quer2 = insrt_quer2 + '%(' + keys[a] + ')s'
        else:
            insrt_quer1 = insrt_quer1 + ',' + keys[a]
            insrt_quer2 = insrt_quer2 + ', %(' + keys[a] + ')s'
    insrt_quer = insrt_quer1 + insrt_quer2 + insrt_quer3
    conn = psycopg2.connect(host = host_name, port = port_name, database = db_name, user = user_name, password = password_name)
    cur = conn.cursor()
    cur.executemany(insrt_quer, fnl_upl)
    conn.commit()
    cur.close()
