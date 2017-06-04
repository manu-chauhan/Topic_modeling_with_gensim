import os
import configparser
from Definitions import ROOT_DIR

author = 'Manu Chauhan'

config = configparser.RawConfigParser()
config_file = os.path.join(ROOT_DIR, 'conf/config.properties')
config.read(config_file)


def get_db_details():
    """
    used for retrieving db related details from config file
    :return: dict of db details
    """
    if not hasattr(get_db_details, "db_detail_dict"):
        get_db_details.db_detail_dict = dict(config.items('Db_Section'))

    return get_db_details.db_detail_dict


def get_sql_dict():
    """
    used for retrieving sqls from config file
    :return: dict of sqls
    """
    if not hasattr(get_sql_dict, "sql_dict"):
        get_sql_dict.sql_dict = dict(config.items('Sql_Section'))

    return get_sql_dict.sql_dict


def get_lda_params():
    """
    used for retrieving lda parameters from config file
    :return: dict of lda parameters
    """
    if not hasattr(get_lda_params, 'param_dict'):
        get_lda_params.param_dict = dict(config.items('Parameters'))

    return get_lda_params.param_dict
