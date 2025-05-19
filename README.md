# revenue_modeling

updated 19-05-2025

Use FLEXBALANCE SAVINGS CALCULATOR online:
https://fb-savings.streamlit.app/

To run locally:

- install requirements
- navigate to folder
- execute "streamlit run flexbalance_savings_streamlit.py"

TO RUN SPOTON BUSINESS CALCULATION:
dependencies:

numpy
pandas


1. navigate to correct folder
2. place input files in the same folder as a csv file:
 a. dayahead prices:
      https://docs.google.com/spreadsheets/d/13pz53-Iu-ecrRw2Li31SrtVyriLvKN_J/edit?gid=477448899#gid=477448899
   b. mFRR data. Last tab of this document:
       https://docs.google.com/spreadsheets/d/1HMjLWTIXSLJ0xDOd_rWDTTAt7bS0eZWY/edit?gid=492126995#gid=492126995
   
3. From a terminal in same folder:
   python spoton_businesscase_calculation.py
