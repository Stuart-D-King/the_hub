import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

'''
DATA:
Listing Data: Listing ID and 81 characteristics of the listing such as price, description, host response time, host response rate, and more for 3586 listings in Boston.

Calendar Data (same listings): Listing ID, date for each day of the year (9/6/2016-9/5/2017), available (f if the listing is not occupied on the date or t if the listing is occupied on the date), price per night if the listing is occupied on the date

QUESTIONS:
1. Use the calendar data to sum number of nights occupied, for the entire year, per listing ID, and merge with the listing data

2. Create a visualization to show seasonality trends for occupancy in Boston

3. Select 5 neighborhoods in Boston, referencing column A 'neighbourhood', in the listing spreadsheet and use text analysis to extract key characteristics of these neighborhoods from column G 'neighborhood_overview' and/or column E 'Description'. Describe your process and create visualizations to display your findings
'''


def prep_data():
    '''
    Load the two datasets into Pandas dataframes, and call the merge_days_booked() function to update the listings dataframe with the sum of nights occupied for each listing.
    '''
    listings = pd.read_csv('data/ListingsAirbnbScrapeExam.csv')
    calendar = pd.read_csv('data/CalendarAirbnbScrapeExam.csv')
    calendar.drop_duplicates(inplace=True)

    calendar['booked'] = np.where(calendar['available'] == 't', 1, 0)
    calendar['date'] = pd.to_datetime(calendar['date'], infer_datetime_format=True)

    listings_merged = merge_days_booked(listings, calendar)

    return listings_merged, calendar


def merge_days_booked(listings, calendar):
    '''
    Solution to Question #1 - Using the calendar data, sum the number of nights occupied, for the entire year, per listing ID, and merge with the listing data.
    '''
    cal_grps = calendar.groupby(['listing_id'], as_index=False)['booked'].sum()
    cal_grps = cal_grps.rename(columns={'booked':'days_booked'})

    merged_df = pd.merge(listings, cal_grps, left_on='id', right_on='listing_id', how='left')

    merged_df.drop('listing_id', axis=1, inplace=True)

    return merged_df


def process_text(txt):
    '''
    Return cleansed text by tokenizing, stripping stopwords, and lemmatizing the tokens. This function is used prior to performing text analysis and topic modeling.
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str(txt).lower())

    sw = set(stopwords.words('english'))
    stopped_tokens = [i for i in tokens if not i in sw]

    wordnet = WordNetLemmatizer()
    clean_txt = ' '.join([wordnet.lemmatize(i) for i in stopped_tokens])

    return clean_txt


if __name__ == '__main__':
    listings, calendar = prep_data()
