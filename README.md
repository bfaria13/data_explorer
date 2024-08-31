# Data Explorer

### Description

- This repo present an exploration of a book review database (+ metadata for books) as part of a challenge.


### Workflow

1. `00-DataPreparation.ipynb`

- This notebook contains all code to carry out the data preparation

2. `01-DataExploration.ipynb`

- This notebook presents a full pipeline to carry out a pretty funny analysis for a table with text columns.

3. `app.py`

- Data explorer application.

> The user should specify one of the parameters: Title, Authors or Categories, then the app will search for that in the database and return a report with all collected informations.

**Usage example:**
```
$ python app.py -g Humanism --topk 5
```
**Results:**

```
Report for categories = Humanism
Date: 31-08-2024_03:01:12

## Analysis for Humanism ##:
> Main descriptive text:
This book is very dense with facts & figures, and concepts that take me some thinking in order to fully grasp. Don't get me wrong, I love it! But I think it will take me some time to finish reading! An excellent resource for understanding the philisophical worldview today.
> Sentiment Analysis: ('positive', (0.811, 'neutral'))
> Main Points list:
[['philisophical worldview today', 'fully grasp'], ['writing class papers', 'also contains several chapters'], ['pathetic negativism ", mainly', 'irony got quite scant reviews'], ['sometimes cause one', 'agnostic friends'], ['provide good reference material', 'generally little known facts'], ['monotheism actually spur scientific thought', 'still use radio metric dating'], ['public schools', 'american public via national geographic'], ['unknowingly call', 'true account'], ['college class could create', 'half ago ..... whows'], ['using true scientific approaches', 'every single facet'], ['fit two books', 'many great books available'], ['valuable resource available', 'underlies evolutionary teaching'], ['may seem expensive', 'think seriously'], ['well reasoned throughout', 'tightly knit together'], ['current scientific evolutionary bias means', 'sufficient scientific detail'], ['discredited theories without replacement could put evolutionists', 'ethics become dependent upon situations']]

## Analysis for Humanistic ethics ##:
> Main descriptive text:
Author, Ies Spetter, was the beloved Leader of the Riverdale Ethical Society (NYC) and founder of the Riverdale Mental Health Clinic. He was a doctoral student in Holland when WWII began. He survived Auschwitz and Buchenwald.Book illustrates his militant humanism, analyzing the impermissible fanaticism that allows the killing of innocents. Dr. Spetter, from his wealth as counseling, clergyman, and professor of Social Psychology, calls for a nurturing of trust as the fundamental law of life, a nurturing of the fullness of human capacity, without the delusion of perfectionism.Should be of interest to humanists, ethicists, Buddhists, Liberal and neo-orthodox theologians.
> Sentiment Analysis: ('positive', (0.729, 'neutral'))
> Main Points list:
[['riverdale ethical society', 'riverdale mental health clinic']]

...continue (break to allow better visualization here!)
```


```
$ app.py [-h] [-t TITLE] [-a AUTHORS] [-g CATEGORIES] [--topk TOPK]

Program to explore a book evaluation database

options:
  -h, --help            show this help message and exit
  -t TITLE, --Title TITLE
                        Book title.
  -a AUTHORS, --authors AUTHORS
                        Author name.
  -g CATEGORIES, --categories CATEGORIES
                        Genre (categories) name.
  --topk TOPK           Max similar titles/author/genre to locate in.
```

### Repo
```
data_explorer           # root
├── data/               # store raw and processed data
├── notebooks/          # store jupyter notebooks
├── reports/            # store reports and imgs 
├── src/                # store source .py files
└── app.py              # execute automatized analysis
```

***
**[Bruna F Faria](https://www.linkedin.com/in/brunafrancielefaria/)** - PhD in Computing, Data Science and Engineering
***