
from helpers import *
from feature_definitions import *
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

if(__name__ == '__main__'):
    # import the "crash" data
    datafile = "../data/txdot_2010_2017.csv"

def preprocess_data(datafile, source='txdot', verbose=0):
    data = pd.read_csv(datafile,header=7)
    # feature definitions and properties
    featdef = get_feature_defs()
    if(verbose):
        print(list(data.columns))

    # try something interesting - track the "categories" of columns
    colgrps = {
        # relevant info for about intersections
        'intersection' : ['street_name','intersecting_street_name','intersection_related'],
      }
    # preprocessing
    # remove punctuation, then lowercase (src: http://stackoverflow.com/a/38931854)
    def process_cols(df):
      return df.columns.str.replace('[,\s()]+','_').str.lower()
    data.columns = process_cols(data)
    regexpunct = '[,\s()-]+'
    def process_data_punctuation(df):
        for category in list(featdef[featdef.type == 'str'].index):
           df[category] = df[category].str.replace(regexpunct,'_').str.lower()
        for category in list(featdef[featdef.type == 'street'].index):
           df[category] = df[category].str.replace(regexpunct,'_')
        return df
    data = process_data_punctuation(data)
    # special cases
    data.columns = data.columns.str.replace('crash_i_d', 'crash_id')
    # </crash_time>
    # convert integer crashtime to datetime with year
    data['crash_datetime'] = create_datetime_series(data)
    featdef = add_feature(featdef, 'crash_datetime', {'type':'datetime','regtype':'categorical'})
    # todo: plot by year, then by time.
    data['crash_time_dec'] = data.crash_datetime.apply(time_base10)
    featdef = add_feature(featdef, 'crash_time_dec', {'type':'float','regtype':'continuous'})
    featdef.set_value('crash_time_dec', 'origin', 'crash_datetime')
    # todo: figure out how to hist() with grouping by year, then group by time within each year
    # data.crash_time_dec.hist(bins=48)
    # if(showplt): 
    #   plt.show()
    data['crash_time_30m'] = data.crash_datetime.apply(time_round30min)
    featdef = add_feature(featdef, 'crash_time_30m', {'type':'int','regtype':'categorical','origin':'crash_datetime'})
    # </crash_time>
    # convert to 'nan'
    if(1):
        # replace ['No Data','Not Applicable'] with NaN
        data.replace(to_replace='No Data', value=np.nan, inplace=True)
        data.replace(to_replace='Not Applicable', value=np.nan, inplace=True)
        data.replace(to_replace='UNKNOWN', value=np.nan, inplace=True) # intersecting_street_name
        data['speed_limit'].replace(0,-1,inplace=True) # speed_limit -  standardise on '-1'
    # GPS coordinates were initially read in as string because missing entries were called 'No Data'
    data['latitude'] = data['latitude'].astype(float)
    data['longitude'] = data['longitude'].astype(float)
    # process categorical data
    if(1):
        # factorizable data
        # convert 'Wet' 'Dry' to '1' '0'
        data['surface_condition'] = data['surface_condition'].factorize()[0]
        # DOC: rename col http://stackoverflow.com/a/11346337
        data.rename(columns={'surface_condition':'surface_wet'})
        # print number of unique entries
        if(verbose):
            for colname in data.columns:
                print("% 4d : %s" % (len(data[colname].unique()), colname))
        # remove data which is has no importance
        # better to drop cols with all NaN and convert "unimportant" data to NaN
        #  - can't universally decide to drop col just based on uniqueness
        # e.g. all of object_struck is 'Not Applicable' and useless, but if surface_condition had only one value "dry" this would be important
        # ? for colname in data.columns:
        # colname = 'object_struck'
        # if(len(data[colname].unique()) == 1):
        #   print("-I-: dropping %s for having all homogenous values %s", (colname, data[colname].unique()[0]))
        #   data.drop(colname,axis=1,inplace=True)

    # binary consolidate features for data exploration.
    # choose binary categories for categorical data to facilitate certain regression techniques
    # this reduces the dimensionality while maintaining control over the features
    # features from weak binary-categories get discarded, and the binary category is not used beyond data exploration!
    # in theory, only some categories will produce strong features
    # however, expanding all categories with e.g. get_dummies would lead to feature explosion, thus increasing the complexity of the dataset
    # this allows the general direction to be found, i.e. which categories can be ignored due to low correlation
    # the strong categories can then be expanded via get_dummies
    # tradeoff/caveat:
    # 1. if one category has one strong feature but many weak features, this technique can hide that feature (correlation will be low, but not zero)
    #    antidote: with enough time, model can be run with each omitted feature
    # 2. the binary re-classification will incorrectly bias the resulting model, and could potentially hide strong features 
    #    this should absolutely only be used for feature selection
    if(1):
        regexpunct = '[,\s()-]+'

        # Explanation: 'Crash Severity' contains the severity of injuries sustained in the crash
        # able to walk away: binary crash severity - was the person badly injured, or could they "walk away"
        category = 'crash_severity'
        bin_category = (category, "bin_%s" % category)
        data[bin_category[1]] = data[bin_category[0]].str.replace('[,\s()-]+','_').str.lower()
        # group ['non_incapacitating_injury','possible_injury','not_injured'] as 1, group ['incapacitating_injury','killed'] as 0, ignore 'unknown' as np.nan, 
        data[bin_category[1]].replace(['non_incapacitating_injury','possible_injury','not_injured'], 1, inplace=True)
        data[bin_category[1]].replace(['incapacitating_injury','killed'], 0, inplace=True)
        data[bin_category[1]].replace(['unknown'], np.nan, inplace=True)
        # track new feature
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

        # Explanation: 'Day of Week' is often thought of as "work week" + "weekend"
        # 1. combine Mo-Th for week, and Fr-Sun for weekend
        # 2. combine Mo-Fr@1600 for week, and Fr@1600-Sun for weekend
        # STUB - not sure if this one makes sense, Sunday night is not particularly wild

        # Explanation: 'Intersection Related' : categorize intersecton by how defensive a cyclist would have to ride
        # 'intersection' and 'intersection_related' are accidents resulting from intersection. in theory, cyclist could be defensive
        # 'non_intersection', 'driveway_access' are accidents without intersection or from a driveway.
        # non_intersection will likely be fault of motorized vehicle (cyclist could, of course, do a bad turn)
        # although driveway could be considered intersection, e.g. cyclist can slow down and watch for car at every driveway,
        # this is still likely the motorists fault -and- a cyclist cannot reasonably avoid a sideways collision
        # TODO: create category of "reasonable" intersection accidents - permute direction, turn, etc
        # 'not_reported' is to be ignored. only present once, and deciding which way undefinite data goes is a bad idea!
        # ['intersection_related', 'intersection',];['non_intersection', 'driveway_access',];['not_reported']
        # categorisation
        bin_category = 'bin_intersection_related'
        category = 'intersection_related'
        # 1 :
        bin_true = ['intersection_related', 'intersection',]
        # 0 :
        bin_false = ['non_intersection', 'driveway_access',]
        # nan :
        bin_znan = ['not_reported']
        data[bin_category] = data['intersection_related'].str.replace(regexpunct,'_').str.lower()
        data[bin_category].replace(bin_true,  1, inplace=True)
        data[bin_category].replace(bin_false, 0, inplace=True)
        data[bin_category].replace(bin_znan, np.nan, inplace=True)
        # track new feature
        bin_category = (category, "bin_%s" % category)
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

        # Explanation: 'Light Condition' can be reduce to "good visibility", "bad visibility"
        # ['dark_lighted', 'dark_not_lighted', 'dusk', 'dark_unknown_lighting', 'dawn',];['unknown',];['daylight']
        # categorisation
        # 0 :
        bin_category = 'bin_light_condition'
        category = 'light_condition'
        binfalse = ['dark_lighted', 'dark_not_lighted', 'dusk', 'dark_unknown_lighting', 'dawn',]
        #   note: this hides the effects of good lighting at night, but the definition of 'dark, not lighted' is not clear, 
        #   and it would require more investigation to determine which cyclists have adequate lighting
        # 1 :
        bintrue = ['daylight']
        # nan :
        binznan = ['unknown',]
        data[bin_category] = data['light_condition'].str.replace(regexpunct,'_').str.lower()
        data[bin_category].replace(bintrue,  1, inplace=True)
        data[bin_category].replace(binfalse, 0, inplace=True)
        data[bin_category].replace(binznan, np.nan, inplace=True)
        # track new feature
        bin_category = (category, "bin_%s" % category)
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

        # Explanation: Manner of Collision - direction of Units involved
        # motorist fault likelihood higher for "non changing" situations, e.g. if going straight
        # categorisation
        category = 'manner_of_collision'
        bin_category = (category, "bin_%s" % category)
        # 1 :
        bin_true = [
                 'one_motor_vehicle_going_straight',
                 'angle_both_going_straight',
                 'one_motor_vehicle_other',
                 'opposite_direction_one_straight_one_left_turn',
                 'same_direction_one_straight_one_stopped'
                 ]
        # 0 :
        bin_false = [
                 'one_motor_vehicle_backing',
                 'same_direction_both_going_straight_rear_end',
                 'opposite_direction_both_going_straight',
                 'one_motor_vehicle_turning_left',
                 'one_motor_vehicle_turning_right',
                 ]
        # nan :
        bin_znan =  ['']
        data[bin_category[1]] = data[bin_category[0]].str.replace('[,\s()-]+','_').str.lower()
        data[bin_category[1]].replace(bin_true,  1, inplace=True)
        data[bin_category[1]].replace(bin_false, 0, inplace=True)
        data[bin_category[1]].replace(bin_znan, np.nan, inplace=True)
        # track new feature
        featattr = dict(featdef.ix[bin_category[0]])
        featdef = add_feature(featdef, bin_category[1], {'type':'int','regtype':'bin_cat','target':featattr['target']})
        featdef.set_value(bin_category[1], 'origin', bin_category[0])

    return(data,featdef)

# run for testing purposes
if(__name__ == '__main__'):
    print("-I-: Self-Test: txdot_parse.py")
    print("-I-: should not see this from an import!")
    (data,featdef) = preprocess_data(datafile, verbose=1)
    print("-I-: End of Pre-Processing")
    print("-I-: Brief data exploration")

    print(data.head())
    print(data.info())
    if(1):
      data.describe()
      data.hist()
      data.corr().plot() # TODO: seaborn
      plt.show()
    else:
      print("-I-: Skipping...")

    # alternative visualisation
    datapt = data.pivot_table(values=['crash_death_count','crash_incapacitating_injury_count','crash_non-incapacitating_injury_count'], index=['speed_limit','crash_time'])
    print(datapt)


    # inspect features with high covariance
    pairplot_bin_var_list = list(featdef[featdef['pairplot']].index)
    if(0):
        sns.pairplot(data, vars=pairplot_var_list)
        plt.show()

    # list of vars which become dummie'd
    dummies_needed_list = list(featdef[featdef.dummies == 1].index)

    # dummies
    # http://stackoverflow.com/a/36285489 - use of columns
    data_dummies = pd.get_dummies(data, columns=dummies_needed_list)
    # no longer need to convert headers, already done in process_data_punctuation
    pp.pprint(list(data_dummies))

    print("-I-: End of File")


# miscellaneous
'''
# look into dictvectorizer dv.get_feature_names http://stackoverflow.com/a/34194521
'''
'''
pandas tricks
filtering
http://stackoverflow.com/a/11872393
# select data with average_daily_traffic_amount but intersecting_street_name null
# => busy roads without an intersection
data[~data['average_daily_traffic_amount'].isnull() & data['intersecting_street_name'].isnull()]

# select intersection_related == 'Non Intersection' and intersecting_street_name null
# => verify whether intersecting_street_name==null indicates that there is no intersection
# => then only display the columns pertaining to street names
data[(data['intersection_related'] == 'Non Intersection') & data['intersecting_street_name'].isnull()][['street_name','intersecting_street_name','intersection_related']]

data[(data['intersection_related'] == 'Non Intersection') & data['intersecting_street_name'].isnull()][colgrps['intersection']]
'''
# if total person count is needed
#  print("-I-: creating total involved count")
#  individual_counts = [
#    'crash_death_count',
#    'crash_incapacitating_injury_count',
#    'crash_non-incapacitating_injury_count'
#    'crash_not_injured_count',
#    'crash_possible_injury_count',
#  ]
#  total_count = pd.Series(index=mapdf.index,dtype=int)
#  for icount in individual_counts:
#    total_count += mapdf[icount]
##    mapdf['person_number'] = mapdf['person_number'] + mapdf[icount]
