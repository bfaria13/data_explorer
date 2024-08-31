import os 
import argparse
import datetime
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyError)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from itables import to_html_datatable
from src.data_preparing import processing_files, get_target_context
from src.data_exploring import (get_similar_text, 
                                get_text_description,
                                get_sentiment,
                                text_summaryzing,
                                get_wordcloud,
                                export_tables                               
                                )

## arguments
parser = argparse.ArgumentParser(description='Program to explore a book evaluation database')
#parser.add_argument('--assistent', type=str, required=False, 
#                    default=False, help='Iteractive assistent (chat data explorer).')
parser.add_argument('-t', '--Title', type=str, required=False, 
                    help='Book title.')
parser.add_argument('-a', '--authors', type=str, required=False,
                    help='Author name.')
parser.add_argument('-g', '--categories', type=str, required=False,
                    help='Genre (categories) name.')
parser.add_argument('--topk', type=int, required=False, default=5,
                    help='Max similar titles/author/genre to locate in.')

args = parser.parse_args()
args = vars(args)
args = {k: v for k, v in args.items() if v is not None}

coltarget = [x for x in args.keys()][0]
filter_condition = [v for k,v in args.items() if k != 'topk'][0]
  
print(f'Parameters: {args}\nTarget column: {coltarget}, filter by: {filter_condition}')


## fix params
filepath='./data'
outdata='./data/df_prepared.csv'
filenames=['books_data.csv', 'Books_rating.csv']
year=2024
OPENAI_KEY='sk-XXX'


## load/prepare datafile
if not os.path.isfile(outdata):
    print('Processing files...')
    df = processing_files(filepath, filenames, year, save=None)
else:
    print('Reading preprocessed file...')
    df = pd.read_csv(outdata)
    
## agg informations by coltarget
print('Generating aggregated data...')
data = get_target_context(data=df, colname=coltarget, filter_condition=None)

## extract informations
print('Filtering data...')
filtered = get_similar_text(data, coltarget, query=filter_condition, topk=args['topk'])

## extract main points of opinions clusters 
print('Adding essential features...')
report = get_text_description(filtered, colname='text', reduced=None, cols2return=None, kw=2)

## extract the sentiment from main phrases of all clusters
print('Adding sentiment analysis...')
sentiment = []
for text in report['Phrase'].values:
    sentiment.append(get_sentiment(text[0]))
report['Sentiment'] = sentiment

## generate a summary from main keywords of all clusters
try:
    print('Adding LLM summary...')
    summary = {'Sentiment':[], 'Summary': []}
    for txt in report['Keywords'].values:
        text = ' '.join([x for xs in txt for x in xs])
        ss = text_summaryzing(text, size=15, OPENAI_KEY=OPENAI_KEY)
        summary['Sentiment'] = summary['sentiment']
        summary['Summary'] = summary['summary']

    report['Sentiment_LLM'] = summary['Sentiment']
    report['Summary_LLM'] = summary['Summary']
except:
    print('Setup a valid OpenAI key (you should have enough quote available).')

## generate a word cloud
print('Generating word clouds...')
for idx,txt in enumerate(report.Keywords.values):
    text = ' '.join([x for xs in txt for x in xs])
    suffix= ''.join(eval(report[coltarget].values[idx])[0])
    get_wordcloud(text, plot=False, save=f'./reports/wordclouds_{suffix}.png')

## write report
print('Writing report...')
date = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
outname = open(f'./reports/report_{date}.txt', 'w')
outname.writelines(f'Report for {coltarget} = {filter_condition}\n')
outname.writelines(f'Date: {date}')

for idx in range(report.shape[0]):
    row = report.iloc[idx]
    outname.writelines(f'\n\n## Analysis for {eval(row[coltarget])[0]} ##:\n')
    outname.writelines(f'> Main descriptive text:\n{row.Phrase[0]}')
    outname.writelines(f'\n> Sentiment Analysis: {row.Sentiment}')
    if 'Summary_LLM' in row.index:
        outname.writelines(f'\n> Summarying (LLM): {row.Summary_LLM} | Sentiment: {row.Sentiment_LLM}')
    outname.writelines(f'\n> Main Points list:\n{row.Keywords}')

outname.close()

## save table report 
print('Saving table as html...')
outname = f'./reports/table_{coltarget}_{date}.html'
export_tables(report, outname)