
import os
import numpy as np
import pandas as pd
import pickle
import csv
import time
import sys
import math
import random
from collections import defaultdict
import json
import shutil
import bisect
from IPython.display import display, clear_output

nrows_train = 37670294
nrows_test = 2528244
nuser_ids = 1198787
nbookings_train = 3000694


FEAT_DICT = {'channel': 0, 'ci_day': 1, 'ci_dayofweek': 2, 'ci_month': 3,
             'ci_quarter': 4, 'cnt': 5, 'co_day': 6, 'co_dayofweek': 7, 'co_month': 8,
             'co_quarter': 9, 'day': 10, 'dayofweek': 11, 'hotel_cluster': 12, 
             'hotel_continent': 13, 'hotel_country': 14, 'hotel_market': 15, 'hour': 16, 
             'is_booking': 17, 'is_mobile': 18, 'is_package': 19, 'minute': 20, 'month': 21, 
             'orig_destination_distance': 22, 'posa_continent': 23, 'quarter': 24, 'site_name': 25, 
             'srch_adults_cnt': 26, 'srch_children_cnt': 27, 'srch_destination_id': 28, 
             'srch_destination_type_id': 29, 'srch_rm_cnt': 30, 'stay_span': 31, 'user_id': 32, 
             'user_location_city': 33, 'user_location_country': 34, 'user_location_region': 35, 
             '0': 36, '1': 37, '2': 38}

LEAK_FEATS = ('id', 'user_location_city', 'srch_destination_id', 'orig_destination_distance')


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


def encode_key(values):
    try: 
        ret = '+'.join([str(v) for v in values])
    except TypeError:
        ret = str(values)
        
    return ret

def decode_key(key):
    helper = tuple(key.split('+'))
    ret = ()
    for k in helper:
        try: 
            ret += (float(k),)
        except ValueError:
            ret += (k,)
    return ret


def try_float(e):
    try:
        ret = float(e)
    except ValueError:
        ret = e
    return ret


def dd():
    return []


def find_leaked_guess2():
    print('Generating keys...')
																																							    	
        
    helper = [10**16 * el[1] + 10**8 * el[2] + el[3] for el in leak_vals]
    helper = sorted(helper)
    print('Done!')
    
    clusters = defaultdict(dd)
    guesses = defaultdict(dd)
    n_tmp = 0
    
    with open('data/train_augmented.csv') as train_file:
        leggo = csv.reader(train_file)
        premessa = 'Reading train lines...'
        print(premessa)
        
        next(leggo)
        
        start_time = time.time()
        for i, line in enumerate(leggo):
            if i>2: print_processing_status(i, nrows_train, start_time, 
                                            frequency=500000, pre_message=premessa)
        
            now_key = tuple(try_float(line[FEAT_DICT[f]]) for f in LEAK_FEATS[1:])
            now_helper = 10**16 * now_key[0] + 10**8 * now_key[1] + now_key[2]

            now_key = encode_key(now_key)
            
            pos = bisect.bisect_left(helper, now_helper)

            try:
                if now_helper == helper[pos] or now_helper == helper[pos+1]:
                    found = False
                    if len(clusters[now_key]) > 0:
                        for p in clusters[now_key]:
                            if line[FEAT_DICT['hotel_cluster']] == p[0]:
                                p[1] += 1
                                found = True                        
                    if not found:
                        clusters[now_key].insert(0, [line[FEAT_DICT['hotel_cluster']], 1])
            except IndexError:
                try:
                    if now_helper == helper[pos]:
                        found = False
                        if len(clusters[now_key]) > 0:
                            for p in clusters[now_key]:
                                if line[FEAT_DICT['hotel_cluster']] == p[0]:
                                    p[1] += 1
                                    found = True                        
                        if not found:
                            clusters[now_key].insert(0, [line[FEAT_DICT['hotel_cluster']], 1])
                except IndexError:
                    pass
    
            if i % 10000000 == 0 and i>0:
                print('Saving partial result...')
                tmp_filename = 'data/leak/tmp_' + str(int(i/10000000)) + '.pickle'
                with open(tmp_filename, 'wb') as outftmp:
                    pickle.dump(clusters, outftmp)
                del clusters
                print('Done!')
                clusters = defaultdict(dd)
                n_tmp += 1

        tmp_filename = 'data/leak/tmp_' + str(4) + '.pickle'
        with open(tmp_filename, 'wb') as outftmp:
            pickle.dump(clusters, outftmp)
        del clusters
        n_tmp += 1
    

    for c_tmp in range(n_tmp):
        print('Loading partial result...')
        load_tmp_filename = 'data/leak/tmp_' + str(c_tmp+1) + '.pickle'
        with open(load_tmp_filename, 'rb') as inftmp:
            loc_clusters = pickle.load(inftmp)
        print('Done!')

        start_time = time.time()

        for c, el in enumerate(leak_vals):
            print_processing_status(c, 1800000, start_time, frequency=100000, 
                                    pre_message='Converting...(part {} of {})'.format(c_tmp+1, n_tmp))
        
            sortini = sorted(loc_clusters[encode_key((el[1], el[2], el[3]))], 
                             key=lambda pair: pair[1])

            if len(guesses[el[0]]) == 0:
                guesses[el[0]] = [p[0] for p in sortini][:5]
            else:
                for cl in sortini:
                    if cl not in guesses[el[0]]: 
                        guesses[el[0]].insert(0, cl)
        
    return guesses


leak_guesses = find_leaked_guess2()

print('Saving clusters...')
with open('data/leak/leak_guesses.pickle', 'wb') as outf:
    pickle.dump(leak_guesses, outf, -1)
print('Done!!')
print('END')
