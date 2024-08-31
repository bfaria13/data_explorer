import pandas as pd

def check_columns(data, columns, sort=False):
    df_check = pd.DataFrame(zip(data[columns].dtypes.index, 
                                data[columns].dtypes, 
                                data[columns].nunique(), 
                                data[columns].isna().sum(),
                                round(data[columns].isna().sum()/data.shape[0], 3)*100), 
                            columns=['Variavel', 'Tipo', 'Qtde_unicos', 'Qtde_NaN', '%_NaN'])
    if sort:
        return df_check.sort_values('Qtde_NaN').reset_index().drop('index', axis=1)
    else:
        return df_check
    
    
def rm_duplicates(data:pd.DataFrame, verbose:bool=False):
    dupl = data.duplicated().sum()
    if dupl > 0:
        data = data.drop_duplicates() 
        if verbose:
            print(f'Total duplicate rows removed: {dupl}')
            print(f'Total of duplicated lines after drop: {data.duplicated().sum()}')
    else:
        print(f'Without duplicated lines!')
    return data


def format_text(text:str, lenwords=20):
    list_words = text.split()
    lw=lenwords
    for idx in range(len(list_words)):
        if idx == lw:
            list_words.insert(idx,'\n')
            lw +=lenwords
    return ' '.join(list_words).replace(' \n ','\n')


def format_text(text:str, lenwords=20):
    list_words = text.split()
    lw=lenwords
    for idx in range(len(list_words)):
        if idx == lw:
            list_words.insert(idx,'\n')
            lw +=lenwords
    return ' '.join(list_words).replace(' \n ','\n')