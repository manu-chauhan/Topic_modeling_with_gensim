# Topic_modeling_with_gensim

![alt tag](https://3.bp.blogspot.com/-oFzTVKm_ciU/VVOXnaLMnSI/AAAAAAAABKw/gL4Hke4_z-Y/s1600/DataScienceEngineering_TagCloud_800.jpg)


________________________________________________________________________________________________________________________________________


Gensim is Licensed under  GNU LGPLv2.1 which requires modifications to Gensim (if any and if distributed to others) to be open sourced. For  more info : https://radimrehurek.com/gensim/about.html

### What is this repository for? ###

**Quick summary:**
 Topic Modeling on collection of documents or texts fetched from database in python3 using Gensim library.
 The current version writes CSVs as output under 'output' directory in project. One csv for retrieved topics along with corresponding words and another csv containing topic distribution for raw texts (unseen/new documents can be used). Future version will have word cloud representing prominent words for each topic.
 
 **NOTE:** Delete contents of data folder (which will store dictionary and corpus) if texts or documents used for training model are changed. 

**Version : 1.1**
Next version will use word cloud to display prominent words for each topic.

### How do I get set up? ###

 **Summary of set up:** Basic setup (I used Pycharm IDE community version) with required modules installed.
 
 Main_file.py is the main file of the project.
 
 Utils.py contains utility methods such as reading config sections from config.properties.
 
 Configure config.properties file to connect to database and modify Sql query accordingly. (postgres as of now, will add mysql connectivity in later versions)

 **Configuration:** Config file includes sections for database, sql query and LDA model parameters.

 **Dependencies:**  Gensim, psycopg2, nltk and stop_words (use pip to install gensim, psycopg2 and stop_words. Google how to download nltk)

 **Database configuration:** DB_Section in config.properties file under conf directory


