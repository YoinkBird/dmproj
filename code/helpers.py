# helper functions
from feature_definitions import *
from txdot_parse import *

# don't need most of these imports
from sklearn.ensemble import RandomForestClassifier
from sklearn import (metrics, model_selection, linear_model, preprocessing, ensemble, neighbors, decomposition)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import pprint as pp
import re
#import xgboost as xgb

def print_test():
    print("hi")
    return("hi")

# time conversions
# convert integer crashtime to datetime with year
# input: dataframe with year and time (int)
# todo: add month
def create_datetime_series(df):
    if('crash_month' in df):
        print("-E-: function can't handle months yet")
        return False
    return pd.to_datetime(df.apply(lambda x: "%s.%04d" % (x.crash_year,x.crash_time), axis=1),format="%Y.%H%M")
# convert to 24h time
# data.crash_time = data.crash_time.apply(lambda x: str(x).zfill(4)) # leading zeros
# could convert to datetime, but this forces a year,month,day to be present
# pd.to_datetime(data.crash_time.apply(lambda x: "2015%s"%x),format="%Y%H%M") # http://strftime.org/
# data.apply(lambda x: "%s%s" % (x.crash_year,x.crash_time), axis=1) # flexible year
# data['datetime'] = pd.to_datetime(data.crash_time.apply(lambda x: "2015%s"%x),format="%Y%H%M")
# src: http://stackoverflow.com/a/32375581
# pd.to_datetime(data.crash_time.apply(lambda x: "2015%s"%x),format="%Y%H%M").dt.time
# final:
# convert to decimal time
# src: https://en.wikipedia.org/wiki/Decimal_time#Scientific_decimal_time
# convert hours to fraction of day (HH/24) and minutes to fraction of day (mm/24*60), then add together
def time_base10(time):
    import pandas as pd
    time = pd.tslib.Timestamp(time)
    dech = time.hour/24; decm = time.minute/(24*60)
    #print("%s %f %f %f" % (time.time(), dech, decm, dech+decm))
    base10 = dech+decm
    return base10
def time_base10_to_60(time):
    verbose = 0
    # only round on final digit
    hours10 = time * 24  # 0.9 * 24  == 21.6
    hours10 = round(hours10, 5) # round out floating point issues
    hours24 = int(hours10)  # int(21.6) == 21
    min60 = round((hours10 * 60) % 60)     # 21.6*60 == 1296; 1296%60 == 36
    if(verbose):
        print("time: %f | hours24 %s | hours10 %s | min60 %s" % (time,hours24,hours10,min60))
    return hours24 * 100 + min60
# round to half hour
def time_round30min(pd_ts_time):
    import datetime
    pd_ts_time = pd.tslib.Timestamp(pd_ts_time)
    newtime = datetime.time()
    retmin = 61
    if(pd_ts_time.minute < 16):
        newtime = datetime.time(pd_ts_time.hour,0)
        retmin = 00
    elif((pd_ts_time.minute > 15) & (pd_ts_time.minute < 46)):
        newtime = datetime.time(pd_ts_time.hour,30)
        retmin = "30"
    elif(pd_ts_time.minute > 45):
        pd_ts_time += datetime.timedelta(hours=1)
        newtime = datetime.time(pd_ts_time.hour,00)
        retmin = 00
    #print("%s %s %f %f" % (pd_ts_time.pd_ts_time(), newtime, newtime.hour, newtime.minute))
    time_str = "%s.%02d%02d" % (pd_ts_time.year, newtime.hour, newtime.minute)
    # omit - would have to specify the year
    # time2 = pd.tslib.Timestamp("%02d:%02d" % (newtime.hour, newtime.minute))
    if(0):
        time2 = pd.to_datetime(time_str, format="%Y.%H%M")
    else:
        time_str = "%02d%02d" % (newtime.hour, newtime.minute)
        time2 = int(time_str)
    return time2

'''
approach:
    0. work with pre-processed data (txdot_parse.preprocess_data)
    1. identify all entries for intersections (street_name, intersecting_street_name) with no speed_limit
        => data2
    2. get all available data for these intersections
        => df3
    3. 
'''
'''
profiling:
    IPython CPU timings (estimated):
      User   :      70.87 s.
      System :       0.12 s.
    Wall time:      72.95 s.
'''
# assume already processed
def impute_mph(data, verbose=0):
    verbose3 = 0
    if(verbose):
        print("-I-: using verbosity %d" % (verbose))
    colgrps = {
        # relevant entries for intersections
        'intersection' : ['street_name','intersecting_street_name'],
        'inter_mph' : ['speed_limit','crash_year','street_name','intersecting_street_name'],
        # speed limit changes over the years
        'inter_mph_uniq' : ['speed_limit','crash_year','street_name','intersecting_street_name'],
      }
    # impute missing speed limits

#    # standardise on '-1' for missing limit
#    data['speed_limit'].replace(0,-1,inplace=True)
#    data['speed_limit'] = data['speed_limit'].replace(0,-1)
    # handled already in txdot_parse
    data.intersecting_street_name.replace('UNKNOWN',np.nan,inplace=True)

    if(verbose):
        print("total missing speed limit data:\n %s" % (data[data['speed_limit'] == -1].shape[0]))
    # df of all intersections and relvant data - keep only attributes which identify unique intersections and speed limits
    df_inter = data[(~ data.intersecting_street_name.isnull())][colgrps['inter_mph_uniq']]
    num_inter = df_inter.shape[0]
    if(verbose):
        print("total intersections:\nin : %s\nout: %s" % (data.shape[0], df_inter.shape[0]))
    df_inter.drop_duplicates(subset=colgrps['inter_mph_uniq'], inplace=True)
    if(verbose):
        print("deduped:\nout: %s" % (df_inter.shape[0]))

    # df of intersections without speed limits, to be used as an "index" to examine each intersection
    df_inter_nomph = df_inter[(df_inter.speed_limit == -1)][colgrps['intersection']].drop_duplicates()
    if(verbose):
        print("intersections without speed_limit:\nout: %s" % (df_inter_nomph.shape[0]))
    df_inter_nomph.reset_index(drop=True, inplace=True)

    # TODO: include crash id
    # data[(data.street_name == ser['street_name']) & (data.intersecting_street_name == ser['intersecting_street_name'])][colgrps['inter_mph']]
    # loop through list of intersections with a missing entry for speed_limit
    # get all entries for these intersections
    # impute
    # TODO: is this a case for a pivot_table ?
    for i,ser in df_inter_nomph.iterrows():
      if(verbose > 1):  #verbose>=2
          print("%d / %d ..." % (i, df_inter_nomph.shape[0]))
      #if (i != 3): continue # skip first 2, they are confirmed boring
      if(verbose3):
          print("%s | %s | %s" % (i, ser.street_name, ser.intersecting_street_name))
      #print(data[(data['street_name'] == ser['street_name'])]) 
      # sometimes the updates to 'dftmp' happen to 'data', something about slices vs copies, blah
      # A value is trying to be set on a copy of a slice from a DataFrame
      # See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      dftmp = data[(data['street_name'] == ser['street_name']) & (data['intersecting_street_name'] == ser['intersecting_street_name'])]
      if(verbose3):
          print("before:")
          print(dftmp['speed_limit'].unique())
          print(dftmp[colgrps['inter_mph']])
      dftmp['speed_limit'].replace(-1,np.nan,inplace=True)
      # sequence: fill back (misses a missing "last" entry), then forward (in order to get the missing "last" entry
      # backwards fill - BIAS towards future speed limits (often higher!)
      dftmp['speed_limit'].fillna(method='bfill',inplace=True)
      # forwards fill - BIAS towards past speed limits (often lower!)
      dftmp['speed_limit'].fillna(method='ffill',inplace=True)
      if(verbose3):
          print("after:")
          print(dftmp[colgrps['inter_mph']])
      # write
      for jq in dftmp.index:
        if(verbose3):
            print("index %d" % jq)
            print(data.ix[jq].speed_limit)
        tmplim = dftmp.ix[jq].speed_limit
        if(np.isnan(data.ix[jq].speed_limit) | (data.ix[jq].speed_limit == -1) ):
            if(not np.isnan(tmplim) ):
                data.set_value(jq,'speed_limit',dftmp.ix[jq].speed_limit)
        if(verbose3):
            print(data.ix[jq].speed_limit)
      #if (i == 6): break # quit after 6 loops
    if(verbose):
        print("total new missing speed limit data:\n %s" % (data[data['speed_limit'] == -1].shape[0]))
    return data

  # dftmp['speed_limit'].replace(-1,np.nan, inplace=True)


if(__name__ == '__main__'):
    test_impute_mph = 1
    test_timeconversion = 0
    # testing - visual inspection
    if(test_timeconversion):
        print("verify correct operation of time_base10")
        # not testing 24:00 -> 1.0 because "hour must be in 0..23" for dateutil
        testtimes1 = ["0:00", "4:48"  , "7:12"  , "21:36" , "23:59"     , "0:59"      , "23:00"    ] # "24:00"
        testtimes2 = [0.0   , 0.2     , 0.3     , 0.9     , 0.999305556 , 0.040972222 , 0.958333333] # 1.0
        for i, testtime in enumerate(testtimes1):
            rettime = time_base10(testtime)
            status = "FAIL"
            # round for comparisson because floating point gets messy
            if(round(testtimes2[i],4) == round(rettime,4)):
                status = "PASS"
            print("%s: %6s: %s == %s ?" % (status, testtime , testtimes1[i] , rettime))
        print("verify correct operation of time_base10_to_60")
        for i, testtime in enumerate(testtimes2):
            status = "FAIL"
            rettime = time_base10_to_60(testtime)
            if(int(testtimes1[i].replace(':','')) == rettime):
                status = "PASS"
            print("%s: %6f: %s == %s ?" % (status, testtime , testtimes1[i] , rettime,))
    if(test_timeconversion):
        print("verify correct operation of time_round30min")
        testtimes1 = ["0:00" , "0:14" , "0:15" , "0:16", "0:29","0:30","0:31","0:44","0:45","0:46", "4:48"  , "7:12"  , "21:36" , "23:59"]
        testtimes2 = ["0:00" , "0:00" , "0:00" , "0:30", "0:30","0:30","0:30","0:30","0:30","1:00", "5:00"  , "7:00"  , "21:30" , "00:00"]
        for i, testtime in enumerate(testtimes1):
            #rettime = time_round30min(pd.tslib.Timestamp(testtime))
            rettime = time_round30min(testtime)
            status = "FAIL"
            if(int(testtimes2[i].replace(':','')) == rettime):
                status = "PASS"
            print("%s: %6s: %s == %s ?" % (status, testtime , testtimes2[i] , rettime))
    if(test_impute_mph):
        # import the "crash" data
        datafile = "../data/txdot_2010_2017.csv"
        (data,featdef) = preprocess_data(datafile, verbose=0)
        totalmissing   = data[data['speed_limit'] == -1].shape[0]
        missingpercent = totalmissing / data.shape[0]
        print("pre : total missing speed limit data:\n %s (%s of 1)" % (totalmissing, missingpercent))
        print(data.speed_limit.unique())
        data = impute_mph(data, verbose=0)
        totalmissing   = data[data['speed_limit'] == -1].shape[0]
        missingpercent = totalmissing / data.shape[0]
        print("post: total missing speed limit data:\n %s (%s of 1)" % (totalmissing, missingpercent))
        print(data.speed_limit.unique())
