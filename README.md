## Boston, The Center of the Universe

## Airbnb Data Challenge
Prepared by Stuart King - October 2017
Objectives:
1. Create a new variable of the number of bookings per listing
2. Create a visualization to show seasonality trends for occupancy in Boston
3. Perform text analysis to extract key neighborhood characteristics

### Table of Contents
- Background & Data
- Total Bookings by Listing
- Occupancy Trends
- Text Analysis
  * Tf-Idf
  * KMeans & Tf-Idf
  * Latent Dirichlet Allocation
- Next Steps

### Background & Data
We are given two data files scraped from Airbnb:
* A `listings` dataset that contains a listing's ID and 81 characteristics such as price, description, host response time, host response rate, and more for 3,586 listings in Boston.

* A `calendar` dataset that contains a listing's ID, a date for each day of the year (September 6, 2016 - September 5, 2017), daily availability ('f' if the listing is not occupied on the date or 't' if it is occupied on the date), and the price per night if the listing is occupied.  

Using the provided data, we must complete the following:
1. Using the `calendar` data, sum the number of nights occupied per listing for the entire year, and merge with the listing data.

2. Create a visualization to show seasonality trends for occupancy in Boston.

3. Select 5 neighborhoods in Boston, referencing the **neighborhood** column in the `listing` dataset, and use text analysis to extract key characteristics of these neighborhoods by using the **neighborhood_overview** and/or **description** columns. Describe your process and create visualizations to support your findings.

### Total Bookings by Listing
