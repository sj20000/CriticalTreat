

# Reference  - accessed 11 March 2023
# - https://github.com/microsoft/mimic_sepsis
# - https://github.com/microsoft/mimic_sepsis/blob/main/preprocess.py

# In[18]:


import argparse
import os

import pandas as pd
import psycopg2 as pg


# sqluser = getpass.getuser()
import psycopg2
from urllib.parse import urlparse


conn = psycopg2.connect(database="mimic",
                        user='postgres', password='Mypostgres1', 
                        host='localhost', port='5432')
print(conn)


# query_schema ='SET search_path to public,mimiciii;'


# Extraction of sub-tables
# There are 43 tables in the Mimic III database. 
# 26 unique tables; the other 17 are partitions of chartevents that are not to be queried directly 
# See: https://mit-lcp.github.io/mimic-schema-spy/

