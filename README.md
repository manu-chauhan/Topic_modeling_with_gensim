# Topic_modeling_with_gensim

Copyright (C) 2017 Manu Chauhan (manuchauhan1992@gmail.com)

Distributed under GNU Affero General Public License 3.0 or later

Gensim is Licensed under  GNU LGPLv2.1 which requires modifications to Gensim (if any and if distributed to others) to be open sourced. For  more info : https://radimrehurek.com/gensim/about.html

## Topic Modeling with Gensim library in Python3 ##


### What is this repository for? ###

**Quick summary:**
 Topic Modeling on collection of documents or texts in python using Gensim library.
 The current version writes CSVs as output under 'output' directory in project. One for retrieved topics along with corresponding words and another csv containing topic distribution for raw texts (unseen documents can be used)
 
 NOTE: Delete contents of data dir (which will store dictionary and corpus) if texts or documents changes on which topic modeling is performed. 

**Version : 1.1**

### How do I get set up? ###

 **Summary of set up:** Basic setup (BTW I used Pycharm IDE community version) with required modules installed.
 
 Main_file.py is the main file of the project.
 Utils.py contains utility methods such as reading config sections from config.properties.
 
 Configure config.properties file to connect to database and modify Sql query accordingly.

 **Configuration:** Config file includes sections for database, sql query and LDA model parameters.

 **Dependencies:**  Gensim, psycopg2, nltk and stop_words (use pip to install gensim, psycopg2 and stop_words. Google how to download nltk)

 **Database configuration:** DB_Section in config.properties file under conf directory
