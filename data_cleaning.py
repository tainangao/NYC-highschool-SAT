import pandas as pd
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]
data = {}
for f in data_files:
    d = pd.read_csv("datasets/{0}".format(f))
    key_name = f.replace(".csv", "")
    data[key_name] = d

all_survey = pd.read_csv('datasets/survey_all.txt', delimiter='\t', encoding='windows-1252')
d75_survey = pd.read_csv('datasets/survey_d75.txt',delimiter="\t", encoding="windows-1252")
survey = pd.concat([all_survey,d75_survey],axis=0)

# rename 'dbn' into 'DBN'
survey["DBN"] = survey["dbn"]

# the survey dataset has over 2000 columns and we'll only keep the necessary ones
survey_fields = [
    "DBN",
    "rr_s",
    "rr_t",
    "rr_p",
    "N_s",
    "N_t",
    "N_p",
    "saf_p_11",
    "com_p_11",
    "eng_p_11",
    "aca_p_11",
    "saf_t_11",
    "com_t_11",
    "eng_t_11",
    "aca_t_11",
    "saf_s_11",
    "com_s_11",
    "eng_s_11",
    "aca_s_11",
    "saf_tot_11",
    "com_tot_11",
    "eng_tot_11",
    "aca_tot_11",
]
survey = survey.loc[:,survey_fields]

# Assign the dataframe survey to the key survey in the dictionary data
data["survey"] = survey

# rename 'dbn' into 'DBN'
data['hs_directory']['DBN']=data['hs_directory']['dbn']

# DBN is a combination of the CSD and SCHOOL CODE columns
# sample DBN: 01M292
# sample SCHOOL CODE: M015
# sample class_size['CSD']: 1
# class_size['CSD'] needs to be padded into 2 digits
def pad_csd(num):
    num=str(num)
    if len(num)==2:
        return num
    else:
        return num.zfill(2)

data['class_size']['padded_csd'] = data["class_size"]["CSD"].apply(pad_csd)
data['class_size']['DBN'] = data['class_size']['padded_csd']+data['class_size']['SCHOOL CODE']

# convert data types
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")
# create a new column that sums 3 SAT scores
data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]
print(data['sat_results']['sat_score'].head())


# get the coordinate for each school
import re
def find_lat(loc):
    coords = re.findall("\(.+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat
data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)

def find_lon(loc):
    coords = re.findall(r'\(.+\)',loc)
    lon = coords[0].split(',')[1].replace(')','')
    return lon
data['hs_directory']['lon'] = data['hs_directory']['Location 1'].apply(find_lon)

data['hs_directory']['lat'] = pd.to_numeric(data['hs_directory']['lat'], errors='coerce')
data['hs_directory']['lon'] = pd.to_numeric(data['hs_directory']['lon'], errors='coerce')



# condense the class_size, graduation, and demographics data sets so that each DBN is unique

### data['class_size']
# each school has multiple values for GRADE, PROGRAM TYPE, CORE SUBJECT (MS CORE and 9-12 ONLY), and CORE COURSE (MS CORE and 9-12 ONLY)
class_size = data["class_size"]
# keep high school only
class_size = class_size[class_size['GRADE ']=='09-12']
# keep GEN ED, the largest PROGRAM TYPE
class_size = class_size[class_size['PROGRAM TYPE']=='GEN ED']

# DBN still isn't completely unique.
# this is due to the CORE COURSE (MS CORE and 9-12 ONLY) and CORE SUBJECT (MS CORE and 9-12 ONLY) columns
# they pertain to different kinds of classes and we need to average them
import numpy as np
class_size = class_size.groupby('DBN').agg(np.mean)
# make DBN a column again
class_size.reset_index(inplace=True)
data['class_size']=class_size
print(data['class_size'].head())

### data['demographics']
# the only column that prevents a given DBN from being unique is schoolyear.
# We only want to select rows where schoolyear is 20112012.
# This will give us the most recent year of data, and also match our SAT results data.
data["demographics"] = data["demographics"][data["demographics"]['schoolyear']==20112012]

### data['graduation']
# the Demographic and Cohort columns are what prevent DBN from being unique
# pick data from the most recent Cohort available, which is 2006
data['graduation'] = data['graduation'][data['graduation']['Cohort']=='2006']
# We also want data from the full cohort, so we'll only pick rows where Demographic is Total Cohort
data['graduation'] = data['graduation'][data['graduation']['Demographic']=='Total Cohort']

###data['ap_2010']
# AP exams have a 1 to 5 scale; 3 or higher is a passing score
# convert data types
cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']
for c in cols:
    data['ap_2010'][c]=pd.to_numeric(data['ap_2010'][c], errors='coerce')



# merge data sets
# let data["sat_results"] be the main data set
combined = data["sat_results"]
# ap_2010 and the graduation data sets have many missing DBN values, so we'll use a left join
combined = combined.merge(data['ap_2010'], how='left',on='DBN')
combined = combined.merge(data['graduation'], how='left',on='DBN')
# these files contain information that's more valuable to our analysis and also have fewer missing DBN values,
# we'll use the inner join type
combined = combined.merge(data['class_size'], how='inner', on='DBN')
combined = combined.merge(data['demographics'], how='inner', on='DBN')
combined = combined.merge(data['survey'], how='inner', on='DBN')
combined = combined.merge(data['hs_directory'], how='inner', on='DBN')

# Fill in any missing values in combined with the means of the respective columns
means = combined.mean()
combined = combined.fillna(means)
# Fill in any remaining missing values in combined with 0
combined = combined.fillna(0)


# build a new column for school district
combined['school_dist']=combined['DBN'].str[0:2]

# delete duplicate columns
combined.drop(columns=['SchoolName', 'School Name', 'SCHOOL NAME',
                       'dbn',
                       'grade1','grade2','grade3','grade4','grade5','grade6','grade7','grade8'], inplace=True)


# export data
combined.to_csv('./cleaned_data.csv', header=True, index=False)