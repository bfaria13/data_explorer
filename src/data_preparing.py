'''
Processing raw data to future exploration
'''

import os
import pandas as pd

from .utils import rm_duplicates

## Loading & Merging datafiles
def loading_merging(filepath:str, filenames:list):    
    df1 = pd.read_csv(os.path.join(filepath, filenames[0]))
    df2 = pd.read_csv(os.path.join(filepath, filenames[1]))

    cols = ['Title', 'authors', 'publisher', 'publishedDate', 'categories', 'score', 'summary', 'text']

    df1_ = df1[[x for x in df1.columns if x in cols]]
    df2_ = df2[[x for x in df2.columns if x in cols]]
    
    return pd.merge(df1_, df2_, on=["Title"], how='left')     

## Treating file
def treating_data(data:pd.DataFrame, YEAR:int):
    ### remove duplicates
    df = rm_duplicates(data) 

    ### create year column based on publishedDate
    mapping_date = {}
    for d in df.publishedDate.unique():
        try:
            mapping_date[d] = int(pd.to_datetime(d).strftime('%Y'))
        except:
            mapping_date[d] = -1
            
    df['publishedDate'] = df['publishedDate'].map(mapping_date)
    df = df.rename(columns={'publishedDate': 'publishedYear'})
    df = df.sort_values(by='publishedYear', ascending=True)
    
    ### create a recency coumn based on publishedYear
    mapping_date = {k:int(YEAR-k) for k in df.publishedYear.unique() if k != -1}
    df.insert(4, 'published_recency', df['publishedYear'].map(mapping_date))

    ### treat missing values
    df = df.dropna(subset='Title')
    df[['authors', 'categories']] = df[['authors', 'categories']].fillna(value=str(['not informed']))
    df = df.fillna(value='not informed')
    return df
  
## Execute
def processing_files(filepath:str, filenames:list, Year:int, save:str=None):
    data = loading_merging(filepath, filenames)
    data = treating_data(data=data, YEAR=Year)
    if save:
        data.to_csv(save, index=False)
    return data


## Get group data for a target column
def get_target_context(data:pd.DataFrame, colname:str, filter_condition:str=None):
    if filter_condition:
        data = data[data[colname] == filter_condition]
        
    if 'title' in colname.lower():
        ### per title
        tmp = data.groupby(colname).agg(authors=('authors', 'first'),
                                        categories=('categories', 'first'),
                                        publisher=('publisher','first'),
                                        year_first=('publishedYear',lambda x: x[x!=-1].min()),
                                        year_last=('publishedYear',lambda x: x[x!=-1].max()),
                                        min_recency=('published_recency','min'),
                                        score_min=('score','min'),
                                        score_max=('score','max'),
                                        score_mean=('score','mean'),
                                        total_evals=('summary','nunique'),
                                        summary=('summary',list),
                                        text=('text',list)).sort_values(
                                            by=['score_mean','total_evals'], ascending=False).reset_index()
        
    elif colname.lower() == 'authors':        
        ### per authors
        tmp = data.groupby(colname).agg(total_titles=('Title','nunique'),
                                        titles=('Title',list),
                                        total_genre=('categories','nunique'),
                                        genre=('categories',list),
                                        total_publisher=('publisher','nunique'),
                                        publisher=('publisher',list),
                                        year_first=('publishedYear',lambda x: x[x!=-1].min()),
                                        year_last=('publishedYear',lambda x: x[x!=-1].max()),
                                        min_recency=('published_recency','min'),
                                        score_min=('score','min'),
                                        score_max=('score','max'),
                                        score_mean=('score','mean'),
                                        total_evals=('summary','nunique'),
                                        summary=('summary',list),
                                        text=('text',list)).sort_values(
                                            by=['score_mean','total_evals'], ascending=False).reset_index()
                                        
    elif colname.lower() == 'categories': 
        ### per genre
        tmp = data.groupby(colname).agg(total_authors=('authors', 'nunique'),
                                        authors=('authors', list),
                                        total_titles=('Title','nunique'),
                                        titles=('Title',list),
                                        total_publisher=('publisher','nunique'),
                                        publisher=('publisher',list),
                                        year_first=('publishedYear',lambda x: x[x!=-1].min()),
                                        year_last=('publishedYear',lambda x: x[x!=-1].max()),
                                        min_recency=('published_recency','min'),
                                        score_min=('score','min'),
                                        score_max=('score','max'),
                                        score_mean=('score','mean'),
                                        total_evals=('summary','nunique'),
                                        summary=('summary',list),
                                        text=('text',list)).sort_values(
                                            by=['score_mean','total_evals'], ascending=False).reset_index()  
        
    #tmp_ = pd.DataFrame(data[colname].value_counts()).rename(columns={'count':'total_evaluations'})
    return tmp #tmp.merge(tmp_, on=colname)