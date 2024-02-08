import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from difflib import SequenceMatcher
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


class Plotter:
    """
    Docstring for Plotter.

        kwards:
            df :
    """

    def __init__(self,**kwargs):
        self.df = kwargs.get('df')

    def plot_horizontal_bar(self, categorical_column, **kwargs):
        """
        Plots horizontal barplots for a categorical column
        **kwargs:
            df : Pandas DataFrame
            ax or axis :
            fig or figure : figure object
            title : title of the plot
            title_size : default = 18
            subtitle :
            xlabel or x_label
            figsize :

        """

        ax = kwargs.get('ax' and 'axis') if kwargs.get('ax' and 'axis') is not None else plt.gca()
        fig = kwargs.get('fig' and 'figure') if kwargs.get('fig' and 'figure') is not None else plt.gcf()
        w,h = kwargs.get('figsize') if kwargs.get('figsize') is not None else (10,7)
        ax.figure.set_size_inches(w,h)
        self.df = kwargs.get('df')

        ## Change this for different cmaps
        mpl.pyplot.viridis()

        plot_title = kwargs.get('title') if kwargs.get('title') is not None else categorical_column + ' Counts'

        if self.df is None:
            raise ValueError('No DataFrame found. Set Parameter df = datframe.')

        # Value counts
        vc = self.df[categorical_column].value_counts(dropna=True)
        index = vc.keys()
        values = vc.values

        title_size = kwargs.get('title_size') if kwargs.get('title_size') is not None else 18

        bars = ax.barh(index, values)

        # plt.tight_layout()
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        title = ax.set_title(plot_title, pad=20, fontsize=title_size)
        title.set_position([.33, 1])

        # plt.subplots_adjust(top=0.9, bottom=0.1)

        ax.grid(zorder=0)

        self.gradientbars(bars)

        rects = ax.patches

        # Place a label each bar
        for rect in rects:
            # Get X and Y placement of label from rect
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label; change to your liking
            space = 10
            # Vertical alignment for positive values
            ha = 'left'

            # If value of bar is negative: place label to the left of the bar
            if x_value < 0:
                # Invert space to place label to the left
                space *= -1
                # Horizontally align label to the right
                ha = 'right'

            # Use X value as label and format number
            label = '{:,.0f}'.format(x_value)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at bar end
                xytext=(space, 0),          # Horizontally shift label by `space`
                textcoords='offset points', # Interpret `xytext` as offset in points
                va='center',                # Vertically center label
                ha=ha,                      # Horizontally align label differently for positive and negative values
            color = 'red')            # Change label color to white


        x_label = kwargs.get('xlabel' and 'x_label') if kwargs.get('xlabel' and 'x_label') is not None else 'Counts'
        #Set x-label
        ax.set_xlabel(x_label, color='#525252')
        ax.set_xlim(values.min(), values.max()+10)

    def gradientbars(self, bars):
        """
        Used exclusively in plot_horizontal_bar

        """
        grad = np.atleast_2d(np.linspace(0,1,256))
        ax = bars[0].axes
        lim = ax.get_xlim()+ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor('none')
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.imshow(grad, extent=[x+w, x, y, y+h], aspect='auto', zorder=1)
        ax.axis(lim)
