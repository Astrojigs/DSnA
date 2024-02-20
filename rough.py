"""
Patient Id
Entry Id
Hospital Number
Surname
Forename
% predicted FEV1
% predicted FVC
% predicted KCO
% predicted TLCO-SB
Adeno-Description
Admission to operation time
Age
Air leak &gt;7 days-Description
Alcohol excess-Description
Anticoagulation-Description
Any complications-Description
ASA grade-Description
Atypical carcinoid-Description
BMI
Broncheoalveolar-Description
Bronchoscopy-Description
BSA
Cardiac failure-Description
Cause of Death
Centre-Description
Centre Name
Chest Wall /Diaphragmatic Procedures-Description
Cigarettes per day
Consultant Surgeon-Description
COPD-Description
Creatinine
CT-Description
CT-guided biopsy-Description
Date And Time Of Admission
Date And Time Of Discharge From Thoracic Surgery Service
Date And Time Of Surgery
Date of Admission
Date Of Birth
Date of discharge / death
Date of discharge from ITU
Date Of Discharge From Thoracic Surgery Team
Date of first assessment
Date of re-admission to ITU
Date of surgery
Date of surgical referral
Description of other method of tissue diagnosis
Details of other histological diagnosis / further information
Details of other named operations
Details of other organs / systems
Diffusion capacity
Dyspnoea score-Description
EBUS-Description
Endoscopic Procedures  (Not VATS)-Description
First Assistant
First Operator
Gender-Description
GMC number
Grade Of surgical Complication-Description
Haemoglobin
Height
Height (feet)
Height (inches)
Hyperlipidaemia-Description
Hypertension-Description
If Offsite, Please Specify Site-Description
If Other Non Cancer Indicaton, Please Specify
If Other Pleural Procedure, Please Specify
If Other Procedure Performed, Please Specify
If Other Procedure, Please Specify
If Other Surgical Approach, Please Specify
Indication For Surgery-Description
Indication For Surgery - Cancer-Description
Indication For Surgery - Non Cancer-Description
Indication For Surgery - Pleural Disease-Description
Infection-Description
Insulin dependent diabetes-Description
IPPV-Description
Ischaemic heart disease-Description
Large cell-Description
Length of post surgical stay (days)
Length of post surgical stay thoracic team (days)
Lung Resections - All Other Pathologies-Description
Lung Resections - Primary Malignant-Description
MATERTHORACICLOCKED-Description
Measured FEV1
Measured FVC
Mediastinal node strategy-Description
Mediastinal Procedures-Description
Mesothelioma Surgery (Therapeutic)-Description
MRI-Description
Name of Trainee
Named operation
Named Operation (Primary Procedure)-Description
NSCLC-Description
Number of functional segments removed
Operation Centre-Description
Operative priority-Description
Other / further information-Description
Other method of tissue diagnosis-Description
Other Operation(LOCAL)
Other Procedures-Description
Other Procedures (Local)-Description
Pack years
Pathological category
Patient status at discharge-Description
Performance status (ECOG)-Description
Peripheral vascular disease-Description
PET-Description
Pleural Procedures-Description
Post-op % predicted FEV1
Post-op % predicted KCO
Post-op predicted FEV1
Post-op predicted KCO
Post-operative M stage-Description
Post-operative N stage-Description
Post-operative T stage-Description
Pre-operative chemotherapy-Description
Pre-operative M stage-Description
Pre-operative N stage-Description
Pre-operative radiotherapy-Description
Pre-operative T stage-Description
Pre-operative tissue diagnosis made-Description
Previous history of cancer-Description
Previous stroke-Description
Primary organ / system targeted
Procedure type
Readmission within 30 Days-Description
Record owner
Referral to operation time
Referring Hospital-Description
Referring Physician-Description
Resection for primary lung cancer-Description
Return to theatre in the same admission-Description
Second Assistant
Small cell-Description
Smoking history-Description
Squamous-Description
Steroid therapy-Description
Surgical Approach - Incision Type
Surgical resection performed
Surgical strategy
Theraputic Broncoscopy Type
Thoracoscore
Time of Admission
Time of Surgery
Total number of functional pulmonary segments
Tracheal Surgery (Includes Carinal Resection)-Description
Trainee led procedure-Description
Type of lung cancer-Description
Type Of Other Procedure Performed-Description
Type Of Other Procedure Performed (Multi Choice)
Typical carcinoid-Description
Undifferentiated-Description
Urea
Weight
Weight (lbs)
Weight (stone)
Years smoked
Count
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from difflib import SequenceMatcher
import geopandas as gpd
import warnings
'''

NOTE: UTILITIES FOLDER MUST BE IN THE SAME DIRECTORY AS UTILS.PY
'''
pd.options.mode.chained_assignment = None  # default='warn'

## Function: Find columns in a df that contain a specific keyword.
def filter_df(df, kwards, columns, prob = True, threshold=0.5):
    """
    Returns a filtered df.
    Parameters:
        kwards = A string or a list of strings to look for in a column
        columns = (str) A column name or a list of column names that we look for.
        prob = (boolean) True if you want to look for similarity (default probability threshold = 0.5)
    """
    og_df = df
    if type(kwards) == list:
        for k,col in zip(kwards,columns):
            if prob == False: # Find exact matches
                df = df[df[col] == k]

            else: # Find similar matches
                k = k.lower()
                df = df[df[col].notna()]
                df[col] = df[col].apply(lambda x : x.lower())
                df[col] = df[col].apply(lambda x : True if check_similarity(x,k) > threshold else False)
                df = df[df[col] == True]
        return og_df.loc[df.index.tolist(), :]

    else:
        if prob == True:
            k = kwards.lower()
            df = df[df[columns].notna()]
            df[columns] = df[columns].apply(lambda x : True if check_similarity(x.lower(), k) > threshold else False)
            df = df[df[columns] == True]
            return og_df.loc[df[columns].index.tolist(), : ]
        else:
            return df[df[columns] == kwards]



def find_in_columns(keyword, df):
    """
    Returns a list of column names in the dataframe that contain the keyword specified.

    Parameters:
        Keyword = A string type value
        df = A Pandas DataFrame
    """
    l = []
    for column in df.columns:
        if keyword.lower() in column.lower():
            l.append(column)
    return l

def find_in_list(keyword,list_):
    """
    Returns a list of items that match a particular keyword in a given list.

    Parameters:
        Keyword = A string type value
        list_ = A list containing string values
    """
    l = []
    for i in list_:
        if keyword.lower() in i.lower():
            l.append(i)
    return l

## Function: Fixes MRNs or Hospital Numbers to 7 digit str format numbers
def fix_id(x):
    """
    Fixes the MRNs or Hospital Number (assumes 7 numbers in the id).
    Use this function using .apply() method.

    For example:
        df['MRN'].apply(fix_id)
    """
    return str((7-len(str(int(x))))*'0' + str(x))

def check_similarity(a,b):
    """
    Checks the similarity between two strings
    Output: percentage ratio proportional to similarity between two strings
    """
    return round(SequenceMatcher(None, a, b).ratio(),2)

def value_counts(series :'df[column]', dropna=True, is_string=True):
    """
    Returns : Key and values

    Parameters:
        is_string : confirms if column type is string
    output form:
        k, v = utils.value_counts(series = df[column])

    """

    k, v = series.value_counts(dropna=dropna).keys(), series.value_counts(dropna=dropna).values
    k = ['No Value' if type(i) != str else i for i in k]

    return k, v

class Graph:
    """
    Docstring for Plotter.

        kwards:
            df : dataframe

        Methods:
            plot_on_county : (GIS plot)

            plot_horizontal_bar : (bar graph)

            plot_pie : (Pie Chart)



    """

    def __init__(self,**kwargs):
        self.df = kwargs.get('df')
        self.county_df = gpd.read_file('./Utilities/counties.shp')



    def plot_pie(self, column, df, explode = [], ax=None, dropna=True, upto_index=None,**kwargs):
        """
        Plots an aesthetic pie plot

        Parameters:
            column : (str) Column name you'd like to plot
            df : the dataframe
            explode : (list) Separates different part of pie
            ax = default is plt.gca()
            dropna = drop null values? if yes, then True

            upto_index : Specify end index value (exlcudes value at that index)
        """
        k, v = df[column].value_counts(dropna=dropna).keys(), df[column].value_counts(dropna=dropna).values

        if upto_index is not None:
            k = k[:upto_index]
            v = v[:upto_index]
        if explode == []:
            explode = [0 for i in range(len(v))]

        if ax is None:
            ax = plt.gca()

        ax.pie(v, labels=k,
                explode = explode,
                autopct = '%1.1f%%', **kwargs)



    def county_plot(self, df,  county_column, column, operation, ax = plt.gca(), log_scale= False, **kwargs):
        """
        Plots given data on map.

        NOTE: the county column in the dataframe should be with Capital Letters for merging


        Parameters:
            df = dataframe
            ax = axis
            column = (type : str) The name of the column you'd like to visualize
                if column contains string type variables

                if column contains numerical variables
            operation = (type : str)
                    'sum' : when you want to groupby county and sum a column for each county
                    'count' : Groupby county and count the rows for each county
                    'common' : Groupby county and find average (for int or float)
                                and common names (for str type column)

            **kwargs:
                title = (str) Set the title of the plot
                cmap =
                    Sequential2 :
                        ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                          'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                          'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
                    Diverging :
                         ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                          'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

                         For more information on colormaps:
                            https://matplotlib.org/stable/users/explain/colors/colormaps.html

        """

        # Copying the dataframe just to be sfe.
        dummy = df

        # Make a column to count the number of entries in the DataFrame.
        dummy['c'] = [1 for x in range(len(dummy))]

        # Making sure the county column matches on both dfs
        dummy[county_column] = dummy[county_column].apply(lambda x : x.upper().replace('COUNTY',''))


        title = f'{column} across Counties' if kwargs.get('title') is None else kwargs.get('title')

        #group the df w.r.t counties and the desired column
        if operation == 'common':
            # for string:
            if dummy[column].dtype == 'O':
                title = 'Common ' + title
                dummy = dummy.groupby(county_column, dropna=True).agg({column : pd.Series.mode}).reset_index()
                dummy[column] = dummy[column].apply(lambda x : x.any() if type(x) != str else x) # only accounts for list if not a string
            else:
                # Finds average number for each county
                if kwargs.get('title') is None:
                    title = 'Average ' + title
                dummy = round(dummy.groupby(county_column,dropna = True)[column].mean(),2).to_frame().reset_index()

        elif operation == 'sum' or operation == 'total':
            title = 'Total ' + title
            dummy = dummy.groupby(county_column, dropna=True)[column].sum().to_frame().reset_index()
            dummy[column] = dummy[column].fillna(0)
        elif operation == 'count':
            title = 'Count of ' + title
            dummy = dummy.groupby(county_column, dropna=True)['c'].count().to_frame().reset_index()
            dummy[column] = dummy['c']
        # Merge based on county
        dummy['County'] = dummy[county_column]
        m_d = pd.merge(self.county_df, dummy, on='County')

        # Plotting the data
        #Boundary plot
        self.county_df.boundary.plot(ax=ax, linewidth = 0.5)

        #Data plot

        # annotate county with string values
        if m_d[column].dtype == 'O':

            m_d['coords'] = m_d['geometry'].apply(lambda x:x.representative_point().coords[:])
            m_d['coords'] = [coords[0] for coords in m_d['coords']]
            for idx, row in m_d.iterrows():
                ax.text(row.coords[0], row.coords[1], s = row[column],
                horizontalalignment='center',
                bbox = {'facecolor' : 'white',
                'alpha' : 0.9,
                'pad': 2,
                'edgecolor':'none'},
                fontsize = 5)
            if m_d[column].apply(lambda x : len(x)).mean() > 10:
                warnings.warn("String values length > 10. Use Acronyms instead.")

        else:
            if log_scale == False:
                m_d.plot(ax=ax,column=column, legend=True,
                 cmap = 'magma' if kwargs.get('cmap') is None else kwargs.get('cmap'))
            else: # Plot a logarithmic scale
                m_d['Log'] = m_d[column].apply(lambda x : np.log(x))
                m_d.plot(ax=ax, column='Log', legend=True,
                cmap = 'magma' if kwargs.get('cmap') is None else kwargs.get('cmap'))
                title+= ' (${log_e}$ ${scale})$'

        # Don't plot axis
        ax.set_axis_off()

        ax.set_facecolor((0.5,0.5,0.8,1))

        ax.set_title(title)

    def county_names(self, ax=None, **kwargs):
        """
        Plots the boundary of Ireland and county names:

        **kwargs:
            fontsize
            color
            boundary : (default is True) (plots boundary of counties)
        """
        if ax is None:
            ax = plt.gca()

        boundary = True if kwargs.get('boundary') is None else kwargs.get('boundary')

        if boundary is True:
            self.county_df.boundary.plot(ax=ax, linewidth = 0.2, color='black')

        self.county_df['coords'] = self.county_df['geometry'].apply(lambda x:x.representative_point().coords[:])
        self.county_df['coords'] = [coords[0] for coords in self.county_df['coords']]
        for idx, row in self.county_df.iterrows():
            ax.text(row.coords[0], row.coords[1], s = row['County'],
            horizontalalignment='center',
            bbox = {'facecolor' : 'none',
            'alpha' : 0.9,
            'pad': 2,
            'edgecolor':'none'},
            fontsize = kwargs.get('s') if kwargs.get('s') is not None else 7,
            color=kwargs.get('color') if kwargs.get('color') is not None else 'white',
            weight = 'bold')

    def erase_frame(self, ax=None, **kwargs):
        """
        Erases the borders of a given axis
        """
        if ax is None:
            ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def xaxis_off(self, ax=None):
        """
        Makes the x axis label disappear.
        """
        ax.xaxis.set_visible(False)

    def yaxis_off(self, ax=None):
        """
        Makes the y axis label disappear.
        """
        ax.yaxis.set_visible(False)
