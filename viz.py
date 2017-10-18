import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
from evolve_main import prep_data


def monthly_occupancy(calendar):
    '''
    Solution to Question #2 - Create a visualization to show seasonality trends for occupancy in Boston.

    Plot the monthly occupancy totals for Boston to observe monthly trends.
    '''
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    date_sum = calendar.groupby('date', as_index=False)['booked'].agg({'daily_bookings':sum})
    date_sum['month'] = date_sum.date.dt.month

    monthly_groups = date_sum.groupby('month')['daily_bookings'].agg({'monthly_bookings':sum})

    mu = monthly_groups['monthly_bookings'].mean()
    stdev = monthly_groups['monthly_bookings'].std()

    ax.plot(monthly_groups.index.values, monthly_groups.values, color='blue', marker='o', alpha=0.5, linewidth=3)

    ax.axhline(y=mu, color='gray', label='Average')
    ax.axhline(y=mu+1.5*stdev, linestyle='--', color='gray')
    ax.axhline(y=mu-1.5*stdev, linestyle='--', color='gray', label='+/- 1.5 St. Deviations')

    ax.set_xticks(range(1,13))
    ax.set_xticklabels(range(1,13))

    ax.legend(loc='best', fontsize=12)
    # ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Bookings', fontsize=14)
    ax.set_title('Monthly Occupancy in Boston', fontsize=18)

    plt.tight_layout()
    plt.savefig('img/monthly_occupancy.png', dpi=400)
    plt.close()


def time_series(calendar):
    '''
    Supplemental visualization of Boston's occupancy - Plot a line graph of the daily bookings in Boston between September 6, 2016 and September 5, 2017.
    '''
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    date_sum = calendar.groupby('date', as_index=False)['booked'].agg({'daily_bookings':sum})

    mu = date_sum['daily_bookings'].mean()
    stdev = date_sum['daily_bookings'].std()

    ax.plot(date_sum.date.values, date_sum.daily_bookings.values, color='blue')

    ax.axhline(y=mu, color='gray', label='Average')
    ax.axhline(y=mu+1.5*stdev, linestyle='--', color='gray', label='Upper Bound')
    ax.axhline(y=mu-1.5*stdev, linestyle='--', color='gray', label='Lower Bound')

    ax.legend(loc='best', fontsize=14)
    ax.set_ylabel('Bookings', fontsize=14)
    ax.set_title('Daily Boston Occupancy', fontsize=18)

    # plt.tight_layout()
    plt.savefig('img/time_series.png', dpi=400)
    plt.close()


def histogram(calendar):
    '''
    Supplemental visualization of Boston's occupancy - Plot a histogram of normalized daily bookings to observe the distribution's shape. Useful for determining if the distruition follows a Gaussian shape, which is required for certain time-series predictive models.
    '''
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    date_sum = calendar.groupby('date', as_index=False)['booked'].agg({'daily_bookings':sum})

    date_count = calendar.groupby('date', as_index=False)['listing_id'].agg({'num_listings':len})

    merged_df = pd.merge(date_sum, date_count, left_on='date', right_on='date')

    merged_df['occ_rate'] = merged_df['daily_bookings'] / merged_df['num_listings']

    mu = merged_df['occ_rate'].values.mean()
    stdev = merged_df['occ_rate'].values.std()

    ax.hist(merged_df['daily_bookings'].values, color='b', alpha=0.5, bins=20, normed=True)

    density = scs.kde.gaussian_kde(merged_df['daily_bookings'].values)
    x_vals = np.linspace(merged_df['daily_bookings'].values.min(), merged_df['daily_bookings'].values.max(), 100)
    kde_vals = density(x_vals)
    ax.plot(x_vals, kde_vals, 'b-')

    plt.savefig('img/histogram.png', dpi=400)
    plt.close()


if __name__ == '__main__':
    plt.close('all')
    listings, calendar = prep_data()

    monthly_occupancy(calendar)
