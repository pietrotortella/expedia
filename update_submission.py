
import os
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import pickle
import csv
import time
import sys
import math
import random
import seaborn as sns
from collections import defaultdict
import collections
import json
import shutil
import bisect

from IPython.display import display, clear_output


#some numbers that takes a while to compute, and I dont want to run the computation again
nrows_train = 37670294
nrows_test = 2528244
nuser_ids = 1198787
nbookings_train = 3000694

base_path = os.getcwd()
data_path = os.path.join(base_path, 'data/')
udata_path = os.path.join(data_path, 'users/')

with open('data/train_auto_augmented.csv') as inf:
    legge = csv.reader(inf)
    columns = next(legge)

TRAIN_FEAT_DICT = dict(list(zip(columns, range(len(columns)))))

with open('data/test_augmented.csv') as inf:
    legge = csv.reader(inf)
    columns = next(legge)
    
TEST_FEAT_DICT = dict(list(zip(columns, range(len(columns)))))

def count_lines(filename, iszip=False):
    """Counts the number of lines in a file"""
    if iszip:
        with gzip.open(filename) as file:
            for i, line in enumerate(file):
                if i % 2**20 == 0:
                    print('Got to line {:.2f}M'.format(i/(10**6)))
    else:
        with open(filename) as file:
            for i, line in enumerate(file):
                if i % 2**20 == 0:
                    print('Got to line {:.2f}M'.format(i/(10**6)))

    return i+1

def get_magnitude(number):
    """
    A function that returns the order of magnitude of a number.
    unit is the label of the order of magnitude, etodivide the exponent
    to divide the number by in order to obtain the appropriate division.
    """
    magnitude = math.log(number, 10)
    n_unit = int(magnitude / 3)
    if n_unit == 0:
        unit = ''
        etodivide = 0
    elif n_unit == 1:
        unit = 'K'
        etodivide = 3
    elif n_unit == 2:
        unit = 'M'
        etodivide = 6
    elif n_unit == 3:
        unit = 'G'
        etodivide = 9
    else:
        unit = 'E' + str(magnitude)
        etodivide = magnitude
        
    return unit, etodivide


def print_processing_status(current_line, tot_line, s_time, frequency=None, pre_message=None):
    """
    A function that prints to screen the processing status of another function.
    Current_line is the line being processed right now, 
    tot_line the total number of lines to be processed,
    s_time the time the computation started,
    frequency how often the print message should be printed (if None, 1/100 of tot_line),
    pre_message the string that should be printed pefore the status message.
    """
    if frequency is None:
        frequency = int(tot_line / 100)
    message = 'Processing line {0:.1f}{4} (of {1:.1f}{4}). \t Elapsed time: {2:.0f} secs, \t ETA: {3:.0f} secs.' 
    
        
    if current_line == 1:
        print('Processing the first line...')
    elif current_line > 1: 
        if current_line % frequency == 0:
            unit, etodivide = get_magnitude(tot_line)
            loc_time = time.time() - s_time

            clear_output(wait=True)
            if pre_message is not None:
                print(pre_message)
            print(message.format(current_line / (10 ** etodivide), tot_line / (10 ** etodivide), 
                  loc_time, (tot_line / current_line - 1) * loc_time, unit))
            sys.stdout.flush()

def maybestrint(el):
    if el == '':
        return 'NO'
    else:
        try:
            return str(int(float(el)))
        except ValueError:
            return str(el)

def encode_key(values):
    """produces a string to be ued as key in dictionaries.
    when a tuple or a list is given, it joins the elements with a +"""
    try: 
        ret = '+'.join([maybestrint(v) for v in values])
    except ValueError:
        ret = maybestrint(values)
        
    return ret

def decode_key(key):
    """from a string of values separated with +, it returns the correspondig tuple of values"""
    helper = tuple(key.split('+'))
    ret = ()
    for k in helper:
        try: 
            ret += (int(float(k)),)
        except ValueError:
            if k == 'NO':
                ret += ('',)
            else:
                try:
                    ret += (float(k),)
                except ValueError:
                    ret += (k,)
    return ret


def convert_dict_to_df(ds, db):
    """takes two dictionaries of dictionaries, produces dataframes out of them
    and return their concatenation"""
    df_seen = pd.DataFrame.from_dict(ds)
    df_book = pd.DataFrame.from_dict(db)

    return pd.concat((df_seen, df_book), keys=[0.0, 1.0]).fillna(0)

class Simple_Cluster_Counter:
    """
    an object that counts how many lines with given hotel_cluster and is_booking there
    are associated to the possible values of the features feats.
    feats is a tuple of columns of the dataframe.
    counter is a DataFrame, with indices the hotel_cluster and is_booking keys, and
    columns the possible combinations of values of feats.
    """
    def __init__(self, feats):
        self.features = feats
        first = [0.0] * 100 + [1.0] * 100
        second = list(range(100)) + list(range(100))
        
        paired = list(zip(first, second))
        indici = pd.MultiIndex.from_tuples(paired, names=['is_booking', 'cluster_id'])
        self.counter = pd.DataFrame(index=indici)
        
        self.features_position = [TRAIN_FEAT_DICT[f] for f in feats]
        
        self.best_table = pd.DataFrame(index=list(range(100)))
                 
            
    def process_line(self, line, ds_counter, db_counter):
        """
        process one line to a dictionary of dictionary
        """
        f_vals = (line[p] for p in self.features_position)
        f_key = encode_key(f_vals)
        hc_val = float(line[TRAIN_FEAT_DICT['hotel_cluster']])
        isb_val = float(line[TRAIN_FEAT_DICT['is_booking']])        
                
        if isb_val == 0:
            ds_counter[f_key][hc_val] += 1
        else:
            db_counter[f_key][hc_val] += 1
        
    
    def count_file_by_lines(self, filename, nlines):
        
        ds_counter = defaultdict(lambda: defaultdict(lambda: 0))
        db_counter = defaultdict(lambda: defaultdict(lambda: 0))      
        
        with open(filename) as infile:
            legge = csv.reader(infile)
            
            start_time = time.time()
            columns = next(legge)
            
            for i, line in enumerate(legge):
                print_processing_status(i, nlines, start_time, frequency=10000, 
                                       pre_message='Creating counters...')
                self.process_line(line, ds_counter, db_counter)
                
        self.counter = convert_dict_to_df(ds_counter, db_counter)
                
    
    def get_top_10(self, values):
        """
        gets the top 10 hotel_clusters that with the given values.
        it takes into account only those values that have a score of at least WARNING_LIMIT,
        while COEFF_ISBOOKING is the weight to give to bookings versus only seen
        """
        WARNING_LIMIT = 200
        COEFF_ISBOOKING = 8
        
        key = values
        warn = False
        if key in self.counter.columns:
            loc_points = self.counter.ix[0.0][key] + COEFF_ISBOOKING * self.counter.ix[1.0][key]
            
            if loc_points.sum() < WARNING_LIMIT:
                warn = True
                ret = [-1 for i in range(10)]
            else:
                ret = loc_points.sort_values(ascending=False).index[:20]
                
        else:
            warn = True
            ret = [-1 for i in range(20)]
            
        return ret, warn
    
    
    def compute_best_table(self, premess=''):
        """
        computes the best_table, which encode the most recurrent hotel_clusters
        with a given key
        """
        
        start_time = time.time()
        ncol = len(self.counter.columns)
        for i, col in enumerate(self.counter.columns):
            if i>2:
                print_processing_status(i, ncol, start_time, frequency=max(int(ncol/200), 1), pre_message=premess)
            
            new_col = pd.Series(index=list(range(100)))
            
            points, warn = self.get_top_10(col)
            
            if warn:
                new_col = new_col.fillna(0)
            else: 
                for pos, cl in enumerate(points):
                    new_col.ix[cl] = 1 / (pos+1)
                new_col = new_col.fillna(0)
                
            self.best_table[col] = new_col
            
    
    def save_counter_to_file(self, dir_path):
        filename = os.path.join(dir_path, 'counter.csv')
        self.counter.to_csv(filename)
            
    def load_counter_from_file(self, dir_path):
        filename = os.path.join(dir_path, 'counter.csv')
        self.counter = pd.read_csv(filename, index_col=[0, 1])
        
    def save_best_table_to_file(self, dir_path):
        filename = os.path.join(dir_path, 'best_table.csv')
        self.best_table.to_csv(filename)
        
    def load_best_table_from_file(self, dir_path):
        filename = os.path.join(dir_path, 'best_table.csv')
        self.best_table = pd.read_csv(filename, index_col=[0])

class Multi_Cluster_Counter:
    """
    a class that puts together many Simple_Cluster_Counter s and computes them
    at the same time
    """
    def __init__(self, feats_list, name):
        self.name = name
        self.path = os.path.join('data/objs', name)
        self.features_list = feats_list
        self.count = dict()
        for feats in feats_list:
            f_key = encode_key(feats)
            self.count[f_key] = Simple_Cluster_Counter(feats)
        
        self.coeff_index = []
        for feats in self.features_list:
            self.coeff_index += [encode_key(feats)]
        
        self.coeff = pd.Series(index=self.coeff_index)
        self.coeff = self.coeff.fillna(0)
        
        self.user_coeff = defaultdict(self.create_zero_serie)
        
    
    def create_zero_serie(self):
        return pd.Series(index=self.coeff_index).fillna(0)
    
    
    def count_file_by_lines(self, filename, nlines):
        ds_counter = dict()
        db_counter = dict()
        
        for feats in self.features_list:
            ds_counter[encode_key(feats)] = defaultdict(lambda: defaultdict(lambda: 0))
            db_counter[encode_key(feats)] = defaultdict(lambda: defaultdict(lambda: 0))
        
        with open(filename) as infile:
            legge = csv.reader(infile)
            
            start_time = time.time()
            columns = next(legge)
            
            for i, line in enumerate(legge):
                print_processing_status(i, nlines, start_time, frequency=10000)
                
                for feats in self.features_list:
                    f_key = encode_key(feats)
                    self.count[f_key].process_line(line, ds_counter[f_key], db_counter[f_key])
        
        print('Converting dicts to dataframes:')
        for i, feats in enumerate(self.features_list):
            print("converting n. {} of {}".format(i, len(self.features_list)))
            f_key = encode_key(feats)
            self.count[f_key].counter = convert_dict_to_df(ds_counter[f_key], db_counter[f_key])
    
    
    def compute_best_tables(self):
        done_message = ''
        for feats in self.features_list:
            premessa = done_message + 'Computing for the features ' + str(feats) + '...'
            self.count[encode_key(feats)].compute_best_table(premess=premessa)
            done_message += 'Feat ' + str(feats) + 'computed! \n'
    
    
    def train_coefficients_on_file(self, filename, nlines):
        with open(filename) as infile:
            legge = csv.reader(infile)
            start_time = time.time()
            next(legge)
            
            for i, line in enumerate(legge):
                print_processing_status(i, nlines, start_time, frequency=10000)
                

                line_cluster = float(line[TRAIN_FEAT_DICT['hotel_cluster']])
                user_id = int(float(line[TRAIN_FEAT_DICT['user_id']]))

                for feats in self.features_list:
                    feat_key = encode_key(feats)
                    val_key = encode_key((line[TRAIN_FEAT_DICT[f]] for f in feats))
                    
                    line_value = self.count[feat_key].best_table.ix[line_cluster][val_key]
                    
                    self.coeff[feat_key] += line_value
                    self.user_coeff[user_id][feat_key] += line_value
                             
                        
    def train_coefficients_randomly(self, filename, nsamples, nlines):
        NBLOCKS = 1000
        sample_indices = []
        
        start_time = time.time()
        for n in range(NBLOCKS):
            print_processing_status(n, NBLOCKS, start_time, frequency=1, 
                                   pre_message='Preparing indices...')
            indices = list(range(n*int(nlines/NBLOCKS), (n+1)*int(nlines/NBLOCKS), 1))
            random.shuffle(indices)
            indices = indices[:int(nsamples/NBLOCKS)]
            sample_indices += sorted(indices)
        
        with open(filename) as infile:
            legge = csv.reader(infile)
            start_time = time.time()
            next(legge)
            
            samples_seen = 0
            for i, line in enumerate(legge):
                try:
                    next_index = sample_indices[samples_seen]
                except IndexError:
                    next_index = -1
                
                if i != next_index:
                    pass
                else:
                    samples_seen += 1
                    print_processing_status(samples_seen, nsamples, start_time, frequency=10000)


                    line_cluster = float(line[TRAIN_FEAT_DICT['hotel_cluster']])
                    user_id = int(float(line[TRAIN_FEAT_DICT['user_id']]))

                    for feats in self.features_list:
                        feat_key = encode_key(feats)
                        val_key = encode_key((line[TRAIN_FEAT_DICT[f]] for f in feats))

                        line_value = self.count[feat_key].best_table.ix[line_cluster][val_key]

                        self.coeff[feat_key] += line_value
        #                self.user_coeff[user_id][feat_key] += line_value
                        
                        
    
    def save_counter_to_file(self, overwrite=False):
        print('Saving...')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        elif overwrite:
            shutil.rmtree(self.path)
            os.makedirs(self.path)
            
        with open(os.path.join(self.path, 'features_list.pickle'), 'wb') as outfile:
            pickle.dump(self.features_list, outfile)
        
        for feats in self.features_list:
            name = encode_key(feats)
            if not os.path.exists(os.path.join(self.path, name)):
                os.makedirs(os.path.join(self.path, name))
            self.count[name].save_counter_to_file(os.path.join(self.path, name))
            print('Save of {} succesfull!'.format(name))
            
            
    def load_counter_from_file(self, path):
        print('Loading counters...')
        if os.path.exists(path):
            
            for feats in self.features_list:
                folder = os.path.join(path, encode_key(feats))
                try:
                    self.count[encode_key(feats)].load_counter_from_file(folder)
                    print('Counter {} loaded succesfully'.format(feats))
                except FileNotFoundError:
                    print('Directory {} not found, load error!'.format(folder))
            
        else:
            raise FileNotFoundError("Directory {} not fount".format(path))

            
    def save_best_tables(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        for feats in self.features_list:
            name = encode_key(feats)
            if not os.path.exists(os.path.join(self.path, name)):
                os.makedirs(os.path.join(self.path, name))
            self.count[name].save_best_table_to_file(os.path.join(self.path, name))
            print('Saving of best table for {} succesful!'.format(name))
            

    def load_best_tables_from_file(self, path):
        print('Loading best tables...')
        if os.path.exists(path):
            
            for feats in self.features_list:
                folder = os.path.join(path, encode_key(feats))
                try:
                    self.count[encode_key(feats)].load_best_table_from_file(folder)
                    print('Best table for {} loaded succesfully'.format(feats))
                except FileNotFoundError:
                    print('Directory {} not found, load error!'.format(folder))
            
        else:
            raise FileNotFoundError("Directory {} not fount".format(path))
            
    
    def save_coefficients_to_file(self):
        filename = os.path.join(self.path, 'global_coefficients.csv')
        self.coeff.to_csv(filename)
        
        users_filename = os.path.join(self.path, 'users_coefficients.pickle')
        with open(users_filename, 'wb') as outf:
            pickle.dump(self.user_coeff, outf)
        
        
    def load_coefficients_from_file(self):
        filename = os.path.join(self.path, 'global_coefficients.csv')
        self.coeff = pd.read_csv(filename, index_col=[0], header=-1)[1]
        self.coeff = self.coeff.rename('Coefficients')
        
        users_filename = os.path.join(self.path, 'users_coefficients.pickle')
        with open(users_filename, 'rb') as inf:
            self.user_coeff = pickle.load(inf)
            
        print('Load coefficients succesfull!')

maestro_feat = [('user_location_city',), ('srch_destination_id',), 
               ('hotel_country', 'user_location_country'), ('hotel_market', 'stay_span'),
               ('srch_destination_type_id', 'stay_span'), ('ci_month', 'hotel_country'),
               ('ci_month', 'user_location_country'), ('user_location_region', 'stay_span'),
               ('dayofweek', 'hotel_country'), ('dayofweek', 'user_location_region')]

maestro = Multi_Cluster_Counter(maestro_feat, name='maestro')
maestro.load_counter_from_file('data/objs/maestro')
maestro.load_best_tables_from_file('data/objs/maestro/')
maestro.load_coefficients_from_file()

def fix_column_names(df):
    return df.rename(columns=lambda name: encode_key(decode_key(name)))

for k in maestro.count.keys():
    maestro.count[k].counter = fix_column_names(maestro.count[k].counter)
    maestro.count[k].best_table = fix_column_names(maestro.count[k].best_table)


def update_submission(sub_filename, coeff, best_tables):
    found_submission = False
    
    print('Loading previous submission...')
    try: 
        with open(sub_filename, 'r') as sub_f:
            legge_sub = csv.reader(sub_f)
            next(legge_sub)

            for i, line in enumerate(legge_sub):
                last_id = int(line[0])

        print('Last id found: ', last_id)
        found_submission = True
        
    except FileNotFoundError:
        print('Previous submission not found, writing a new one')
    
    
    with open('data/test_augmented.csv') as infile, \
    open(sub_filename, 'a') as outfile:
        
        leggo = csv.reader(infile)
        #scrivo = csv.writer(outfile)
        if not found_submission:
            outfile.write("id,hotel_cluster \n")
        
        next(leggo)
                
        start_time = time.time()
        nogo = 0
        
        printed = False
        for i, line in enumerate(leggo):
            
            now_id = int(float(line[TEST_FEAT_DICT['id']]))
            if now_id <= last_id:
                pass
            else:
            
                points = pd.Series(index=list(range(100))).fillna(0)

                if i>2:
                    print_processing_status(now_id, nrows_test - last_id, start_time, frequency=2000,
                                           pre_message='Writing submission... (nogo = {})'.format(nogo))

                for f_key in best_tables.keys():
                    feats = decode_key(f_key)
                    try:
                        vals = [int(float(line[TEST_FEAT_DICT[f]])) for f in feats]
                        v_key = encode_key(vals)
                        try:
                            points += coeff[f_key] * best_tables[f_key][v_key]

                        except KeyError:
                            nogo += 1
                            pass

                    except ValueError:
                        nogo += 1
                        pass



                top5 = points.sort_values(ascending=False)[:5]
                top5_clusters = list(top5.index)

                towrite = str(i) + ", " + " ".join((str(a) for a in top5_clusters)) + '\n' 

                outfile.write(towrite)

b_tables = dict()
for feats in maestro_feat:
    b_tables[encode_key(feats)] = maestro.count[encode_key(feats)].best_table

update_submission('data/submissions/third_good_try_bis.csv', maestro.coeff, b_tables)