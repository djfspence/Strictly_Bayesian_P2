__author__ = 'David'

#********************************************************************************************************************
#
# Estimation of audience scores in Strictly Come Dancing using MCMC

# 20 November 2018


#********************************************************************************************************************
#
# Further Development

# 1. Stopping criteria.

# MCMC convergence is tricky!
# Appears that multiple runs with varied start points is one way to go

# Gelman-Rubin diagnostic ( )
# Compute m independent Markov chains
# Compares variance of each chain to pooled variance
# If initial states (Theta1j) are overdispersed, then approaches unity from above
# Provides estimate of how much variance could be reduced by running chains
# longer
# It is an estimate!
# http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf

# do as an added dimension in the arrays, run_number
# say 5-10 runs?
# so...start points...I still think these are valid BUT maybe add noise to them and/or have some start points
# that are simply random

# 2. Further series

# Add the data for other series


# 3. Categorised

# Define competitors in terms of their categories and then solve popularity by category
# Also do by individual - some way of finding individual popularity difference to simple category?
# Probably only by ranking within a series?
# Age, gender, ethnicity, slimness, profession (sport, TV, acting, music, politics), parent? (hypothesis -
# attractive women are OK as long as they are mums?), has been a model at some point
# = proxy for physical attractiveness? Seems to crop up with a lot of femal tv presenters
# how to handle the multiple people, former model and singer nor presenting tv show...?
#



# 4. Checking and testing

# Use model to generate a set of fake results from given comp pop values then use the model to get back to those values
# Check the final round is working OK i.e. judges' score does not count
#


# 5. Computing final probability of dance-off

# Given the mean values for competitor popularity - compute the probabilities for each competitor to be in the
# dance off in each round and line up against what actually happenned
# interesting to see how big the divergence is
# good sense check - if it is way off then there is something dodgy happenning


# 6. Change judges score

# Instead of normalising simply make each score in [0,1]
# Worried that later in the competition, with high and almost equal judges scores the normalisation emphasis the
# difference. Maybe not so much an issue for judge's point but it could be an issue when combining with competitor
# popularity.

# 7. Checking - series 11

# Natalie Gumede and Abbey Clancy were both in the dance off but are much more popular than Susanna Reid
# Feels like this is wrong
# Susanna also beat Sophie Ellis B in round 13 on audience scores BUT again SEB looks to be more popular


#********************************************************************************************************************
#
# Imports


import numpy as np
import sys
import random
import pandas as pd
from collections import OrderedDict
from datetime import datetime


#********************************************************************************************************************
#
# Timestamp

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

#********************************************************************************************************************
#
# Key parameters to change


num_chains = 3
block_size = 5
#converged_blocks = 10
burn_in_blocks = 1
converged = False
convergence_target = 1.001
block_number = 0
max_num_blocks = 4
max_iters = max_num_blocks * block_size + 1

#********************************************************************************************************************
#
# Fixed parameters

num_series = 15
max_rounds = 20
max_competitors = 20
num_variable_parameters = 4

# competitors per series plus the overall variable parameters
block_means_array_size = (max_competitors * num_series) + num_variable_parameters

block_means_array = np.zeros((block_means_array_size, max_num_blocks))
block_means_array[:] = np.nan

underflow_avoidance = 1e-100   #add to the base and variant probbailities to avoid inf / zero

#beta distribution (maintains uniform sampling until end of burn-in)
# shape_initial = 2.5
# shape_final = 10
# beta_change_iter_prop = 0.5

#fixed_parameters

num_prob_iters = 10 #number of iterations to compute probability of dance-off
max_judge_score_norm = 1.0

#********************************************************************************************************************
#
# Variable parameters

# the main one!
# popularity of each competitor in the range [0,1]

competitor_series_popularity_iterations = np.zeros((num_chains, num_series, max_competitors, max_iters))
competitor_series_popularity_iterations[:] = np.nan

# variable_parameters_current = np.zeros((num_variable_parameters))
# variable_parameters_current[:] = np.nan
#
# variable_parameters_alternate = np.zeros((num_variable_parameters))
# variable_parameters_alternate[:] = np.nan

variable_parameters_iterations = np.zeros((num_chains, num_variable_parameters, max_iters))
variable_parameters_iterations[:] = np.nan

# log_likelihood_series_current = np.zeros(num_series)  #need to initialise this on first run
# log_likelihood_series_current[:] = np.nan
#
# log_likelihood_series_iterations = np.zeros((num_series, max_iters))
# log_likelihood_series_iterations[:] = np.nan
#
series_log_like_live_array = np.zeros((num_chains, num_series, max_iters))
series_log_like_live_array[:] = np.nan


# just the sum of the above
# log_likelihood_total_current = 0   #need to initialise this on first run
# log_likelihood_total_iterations = np.zeros(max_iters)

# array for holding the convergence values
max_conv_block = np.zeros(max_num_blocks)
max_conv_block[:] = np.nan



#********************************************************************************************************************
#
# Functions

def round_log_likelihood(series, round, competitor_popularity, variable_params):

    '''compute log likelihood of dance-offs for a round given the competitor_popularity and variable-params'''


    #think about this carefully - competitor might miss a week...

    #competitors_in_round = np.where(~np.isnan(competitor_popularity))[0]
    competitors_in_round = series_round_competitor_dict[series][round]

    num_competitors = competitor_popularity.shape[0]

    judge_score_weight = variable_params[0]
    epsilon_counts = int(variable_params[1])
    noise_std_dev = variable_params[2]
    audience_score_round = np.zeros(num_competitors)
    audience_score_round[:] = np.nan
    dance_off_count = np.zeros(num_competitors, dtype=int)

    #normalise the judge score so that it is in line with competitor popularity so judge score weight works ok
    #judge_score_round_norm = normalise_vector_std_normal(judge_score_normalised[series, round, :])

    judge_score_round_norm = judge_score_normalised[series, round, :]

    # audience_score_round is the combination of competitor popularity plus judge's score

    for i in range(num_prob_iters):

        #take the competitor probabilities and add noise to them

        #generate noise vector
        noise_vector = np.random.normal(0.0, noise_std_dev, num_competitors)

        competitor_popularity_plus_noise = normalise_vector_std_normal(competitor_popularity + noise_vector)

        audience_score_round[competitors_in_round] = ((1.0 - judge_score_weight) * competitor_popularity_plus_noise[
            competitors_in_round]) + (judge_score_weight * judge_score_round_norm[competitors_in_round])

        audience_score_round = normalise_vector_std_normal(audience_score_round)

        audience_points = convert_score_to_points(audience_score_round)

        #combine judge score and audience score

        if judge_score_counts[series, round]:
            combined_score = judge_points[series, round, :] + audience_points

        else:
            combined_score = audience_points

        ranked_combined_scores = np.argsort(combined_score)

        #allow for variable number of dance-off positions

        for j in range(num_dancers_dance_off_round[series][round]):
            dance_off_count[ranked_combined_scores[j]] += 1

        #print 'dance_off_competitors', dance_off_competitors
        # dance_off_count[ranked_combined_scores[0]] += 1
        # dance_off_count[ranked_combined_scores[1]] += 1


    #compute probabilities

    dance_off_count += epsilon_counts

    dance_off_prop = dance_off_count / (1.0 * (num_prob_iters + (2 * epsilon_counts)))

    dance_off_prop_cir = dance_off_prop[competitors_in_round]
    comp_in_danceoff = competitor_in_dance_off[series, round, competitors_in_round]

    log_prob = np.sum(np.log(dance_off_prop_cir[comp_in_danceoff == True])) + np.sum(np.log(1.0 - dance_off_prop_cir[comp_in_danceoff == False]))

    return log_prob

def series_log_like(series, competitor_popularity, variable_parameters):

    '''Compute log_likelihood of observed dance-offs for series'''

    series_log_likelihood = 0

    for round in range(max_rounds):
        if valid_series_round[series, round]:

            # then this was a scored round in that series
            round_log_like = round_log_likelihood(series, round, competitor_popularity,
                                                  variable_parameters)
            series_log_likelihood += round_log_like

    return series_log_likelihood


def all_series_log_likelihood(competitor_pop_series, variable_params):

    asll = 0.0

    for series in series_list:

        competitor_pop = competitor_pop_series[series, :]
        sll = series_log_like(series, competitor_pop, variable_params)
        asll += sll

    return asll


def convert_score_to_points(raw_scores):

    '''take an input array of raw scores and return an array of ranked scores (points)'''

    raw_scores[raw_scores == 0] = np.nan

    num_active_competitors = np.count_nonzero(~np.isnan(raw_scores))

    argsort_raw_scores = np.argsort(-raw_scores)
    sorted_raw_scores = -np.sort(-raw_scores)

    point_scores_rank = np.zeros(raw_scores.shape[0])
    point_scores_rank[:] = np.nan

    point_scores = np.zeros(raw_scores.shape[0])
    point_scores[:] = np.nan

    point_scores_rank[0] = num_active_competitors

    for rank in range(1, num_active_competitors):
        if sorted_raw_scores[rank] == sorted_raw_scores[rank-1]:
            point_scores_rank[rank] = point_scores_rank[rank-1]
        else:
            point_scores_rank[rank] = point_scores_rank[rank-1] - 1

    point_scores[argsort_raw_scores] = point_scores_rank

    return point_scores



def normalise_vector_std_normal(vector):

    '''Takes an input vector and normalises it to zero mean and unit std_dev'''

    vector_non_nan = vector[~np.isnan(vector)]
    normalised_vector = (vector - np.mean(vector_non_nan)) / np.std(vector_non_nan)

    return normalised_vector


def create_danceoff_dict(danceoff_bool_array):

    '''Take numpy array of boolean values and create dictionary'''

    danceoff_dict = {}

    for series in range(num_series):
        round_competitor_dict = {}
        valid_series = False
        for round in range(max_rounds):
            competitor_list=[]
            valid_series_round = False
            for competitor in range(max_competitors):
                if danceoff_bool_array[series][round][competitor]:
                    #then this combination is valid
                    competitor_list.append(competitor)
                    valid_series_round = True
                    valid_series = True
            if valid_series_round:
                round_competitor_dict[round] = competitor_list
        if valid_series:
            danceoff_dict[series] = round_competitor_dict

    return danceoff_dict


def create_competitor_lists(judge_scores):

    series_list = []
    valid_series_round = np.full((num_series, max_rounds), False, dtype=bool)
    valid_series_competitor = np.full((num_series, max_competitors), False, dtype=bool)

    for series in range(num_series):
        for round in range(max_rounds):
            for competitor in range(max_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    series_list.append(series)

    series_set = sorted(set(series_list))

    series_rounds_dict = {}
    full_dict = {}

    for series in series_set:
        rounds_list = []
        for round in range(max_rounds):
            for competitor in range(max_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    rounds_list.append(round)
                    valid_series_round[series, round] = True
                    valid_series_competitor[series, competitor] = True
        rounds_set = sorted(set(rounds_list))
        series_rounds_dict[series] = rounds_set


    for series, round_list in series_rounds_dict.items():
        round_competitor_dict = {}
        for round in round_list:
            #print series, round
            competitor_list = []
            for competitor in range(max_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    competitor_list.append(competitor)
            competitor_set = sorted(set(competitor_list))
            round_competitor_dict[round] = competitor_set
        full_dict[series] = round_competitor_dict

    return valid_series_round, valid_series_competitor, full_dict

def sample_beta(mode, shape):

    '''return a sample from a beta distribution of given mode and shape'''

    m = 1.0 * mode
    s = 1.0 * shape

    a = m * (s - 2.0) + 1.0
    b = s - 1.0 - (m * (s - 2.0))

    return np.random.beta(a, b)


def gelman_rubin_diagnostic(input_array):

    '''Compute the Gelman-Rubin diagnostic R_hat'''

    #http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf

    #takes an input array of m chains, n iterations, for p parameters [m,p,n]

    m = input_array.shape[0]
    p = input_array.shape[1]
    n = input_array.shape[2]

    theta_bar = np.mean(input_array, axis = 2)
    sj2 = np.var(input_array, ddof=1, axis = 2, dtype=np.float64)
    theta_2bar = np.mean(theta_bar, axis=0)
    B = n * np.var(theta_bar, ddof=1, axis = 0, dtype=np.float64)
    W = np.mean(sj2, axis=0)
    Var_theta = ((1.0 - (1.0/n))*W) + ((1.0/n)*B)
    R_hat = np.sqrt(np.divide(Var_theta, W))

    return R_hat


#********************************************************************************************************************
#
# Set up data structures

# competitors by series

judge_score = np.zeros((num_series, max_rounds, max_competitors))
judge_score[:] = np.nan

#use max judge score to mormalise judge scores into [0,1]
max_judge_score = np.zeros((num_series, max_rounds))
max_judge_score[:] = np.nan

judge_points = np.zeros((num_series, max_rounds, max_competitors))
judge_points[:] = np.nan

popularity_score = np.zeros((num_series, max_rounds, max_competitors))
popularity_score[:] = np.nan

audience_points = np.zeros((num_series, max_rounds, max_competitors))
audience_points[:] = np.nan

combined_points = np.zeros((num_series, max_rounds, max_competitors))
combined_points[:] = np.nan

competitor_in_dance_off = np.full((num_series, max_rounds, max_competitors), False, dtype=bool)

judge_score_counts = np.full((num_series, max_rounds), True, dtype=bool)

competitor_name_by_series_dict = {}

# =1 if this was an elimination round in the series




# need to do this by import from csv ideally


#********************************************************************************************************************
#
# Series 11 data

#the final (round 13) has two votes so it is considered here as two rounds, 13 and 14


series = 11

competitor_name_dict = {}
competitor_name_dict[1] = 'Abbey Clancy'
competitor_name_dict[2] = 'Susanna Reid'
competitor_name_dict[3] = 'Natalie Gumede'
competitor_name_dict[4] = 'Sophie Ellis-Bextor'
competitor_name_dict[5] = 'Patrick Robinson'
competitor_name_dict[6] = 'Ashley Taylor Dawson'
competitor_name_dict[7] = 'Mark Benton'
competitor_name_dict[8] = 'Ben Cohen'
competitor_name_dict[9] = 'Fiona Fullerton'
competitor_name_dict[10] = 'Dave Myers'
competitor_name_dict[11] = 'Rachel Riley'
competitor_name_dict[12] = 'Deborah Meaden'
competitor_name_dict[13] = 'Julien Macdonald'
competitor_name_dict[14] = 'Vanessa Feltz'
competitor_name_dict[15] = 'Tony Jacklin'

competitor_name_by_series_dict[series] = competitor_name_dict

competitor_position_dict = {}

for i in range(15):
    competitor_position_dict[i+1] = i+1

competitor_position_by_series_dict = {}
competitor_position_by_series_dict[series] = competitor_position_dict


#use max judge score to mormalise judge scores into [0,1]
max_judge_score[11][2] = 80
max_judge_score[11][3] = 40
max_judge_score[11][4] = 40
max_judge_score[11][5] = 40
max_judge_score[11][6] = 40
max_judge_score[11][7] = 40
max_judge_score[11][8] = 40
max_judge_score[11][9] = 40
max_judge_score[11][10] = 40
max_judge_score[11][11] = 46
max_judge_score[11][12] = 80
max_judge_score[11][13] = 80
max_judge_score[11][14] = 120

judge_score[11][2][1] = 62
judge_score[11][3][1] = 28
judge_score[11][4][1] = 35
judge_score[11][5][1] = 35
judge_score[11][6][1] = 34
judge_score[11][7][1] = 36
judge_score[11][8][1] = 37
judge_score[11][9][1] = 32
judge_score[11][10][1] = 40
judge_score[11][11][1] = 40
judge_score[11][12][1] = 78
judge_score[11][13][1] = 78
judge_score[11][14][1] = 116

judge_score[11][2][2] = 65
judge_score[11][3][2] = 36
judge_score[11][4][2] = 35
judge_score[11][5][2] = 35
judge_score[11][6][2] = 38
#judge_score[11][7][2] = did not compete
judge_score[11][8][2] = 39
judge_score[11][9][2] = 37
judge_score[11][10][2] = 36
judge_score[11][11][2] = 41
judge_score[11][12][2] = 78
judge_score[11][13][2] = 79
judge_score[11][14][2] = 119

judge_score[11][2][3] = 59
judge_score[11][3][3] = 34
judge_score[11][4][3] = 29
judge_score[11][5][3] = 32
judge_score[11][6][3] = 34
judge_score[11][7][3] = 36
judge_score[11][8][3] = 39
judge_score[11][9][3] = 31
judge_score[11][10][3] = 33
judge_score[11][11][3] = 33
judge_score[11][12][3] = 70
judge_score[11][13][3] = 73
judge_score[11][14][3] = 112

judge_score[11][2][4] = 64
judge_score[11][3][4] = 31
judge_score[11][4][4] = 35
judge_score[11][5][4] = 30
judge_score[11][6][4] = 28
judge_score[11][7][4] = 32
judge_score[11][8][4] = 34
judge_score[11][9][4] = 31
judge_score[11][10][4] = 36
judge_score[11][11][4] = 38
judge_score[11][12][4] = 71
judge_score[11][13][4] = 74

judge_score[11][2][5] = 55
judge_score[11][3][5] = 27
judge_score[11][4][5] = 33
judge_score[11][5][5] = 28
judge_score[11][6][5] = 34
judge_score[11][7][5] = 37
judge_score[11][8][5] = 35
judge_score[11][9][5] = 32
judge_score[11][10][5] = 38
judge_score[11][11][5] = 41
judge_score[11][12][5] = 69

judge_score[11][2][6] = 57
judge_score[11][3][6] = 31
judge_score[11][4][6] = 31
judge_score[11][5][6] = 31
judge_score[11][6][6] = 33
judge_score[11][7][6] = 35
judge_score[11][8][6] = 35
judge_score[11][9][6] = 35
judge_score[11][10][6] = 35
judge_score[11][11][6] = 37

judge_score[11][2][7] = 46
judge_score[11][3][7] = 26
judge_score[11][4][7] = 26
judge_score[11][5][7] = 28
judge_score[11][6][7] = 25
judge_score[11][7][7] = 23
judge_score[11][8][7] = 29
judge_score[11][9][7] = 28
judge_score[11][10][7] = 29

judge_score[11][2][8] = 44
judge_score[11][3][8] = 28
judge_score[11][4][8] = 31
judge_score[11][5][8] = 27
judge_score[11][6][8] = 32
judge_score[11][7][8] = 26
judge_score[11][8][8] = 32
judge_score[11][9][8] = 27

judge_score[11][2][9] = 46
judge_score[11][3][9] = 28
judge_score[11][4][9] = 22
judge_score[11][5][9] = 30
judge_score[11][6][9] = 28
judge_score[11][7][9] = 26
judge_score[11][8][9] = 29

judge_score[11][2][10] = 33
judge_score[11][3][10] = 16
judge_score[11][4][10] = 23
judge_score[11][5][10] = 17
judge_score[11][6][10] = 19
judge_score[11][7][10] = 20

judge_score[11][2][11] = 47
judge_score[11][3][11] = 27
judge_score[11][4][11] = 26
judge_score[11][5][11] = 22
judge_score[11][6][11] = 30

judge_score[11][2][12] = 48
judge_score[11][3][12] = 28
judge_score[11][4][12] = 23
judge_score[11][5][12] = 27

judge_score[11][2][13] = 38
judge_score[11][3][13] = 22
judge_score[11][4][13] = 23

judge_score[11][2][14] = 42
judge_score[11][3][14] = 20

judge_score[11][2][15] = 29

#judge scores counting (i.e. if not audience score only...)
#defaulted to tru so only need to add the false values

judge_score_counts[11][13] = False
judge_score_counts[11][14] = False

#input dance-offs

competitor_in_dance_off[11][2][13] = True
competitor_in_dance_off[11][2][15] = True

competitor_in_dance_off[11][3][13] = True
competitor_in_dance_off[11][3][14] = True

competitor_in_dance_off[11][4][11] = True
competitor_in_dance_off[11][4][13] = True

competitor_in_dance_off[11][5][5] = True
competitor_in_dance_off[11][5][12] = True

competitor_in_dance_off[11][6][1] = True
competitor_in_dance_off[11][6][11] = True

competitor_in_dance_off[11][7][7] = True
competitor_in_dance_off[11][7][10] = True

competitor_in_dance_off[11][8][7] = True
competitor_in_dance_off[11][8][9] = True

competitor_in_dance_off[11][9][7] = True
competitor_in_dance_off[11][9][8] = True

competitor_in_dance_off[11][10][6] = True
competitor_in_dance_off[11][10][7] = True

competitor_in_dance_off[11][11][5] = True
competitor_in_dance_off[11][11][6] = True

competitor_in_dance_off[11][12][2] = True
competitor_in_dance_off[11][12][5] = True

#one person eliminated directly
competitor_in_dance_off[11][13][4] = True

#i.e. the two runners-up
competitor_in_dance_off[11][14][2] = True
competitor_in_dance_off[11][14][3] = True



#********************************************************************************************************************
#
# Series 14 data

# series 14

# Couple	  Place	    1	2	1+2	3	4	5	6	7	8	9	10	11	12	13
# Ore & Joanne	        1	27	27	54	35	39	36	32	34	35	38	36+4=40	36	38+39=77	39+40+40=119
# Danny & Oti	        2	31	32	63	36	36	35	30	38	38	40	40+5=45	38	37+39=76	36+40+40=116
# Louise & Kevin	        31	32	63	31	33	33	35	39	37	38	39+6=45	37	37+36=73	38+38+40=116
# Claudia & AJ	        4	26	30	56	36	30	32	36	33	36	36	36+3=39	37	35+38=73
# Judge Rinder & Oksana	5	25	27	52	27	27	29	32	33	33	33	29+2=31	31
# Ed & Katya	        6	21	23	44	24	16	18	26	27	25	23	23+1=24
# Greg & Natalie	    7	27	26	53	32	28	24	26	32	31	32
# Daisy & 	        8	32	30	62	31	31	32	33	34	31
# Laura & Giovanni	    9	25	32	57	30	33		36	32
# Anastacia & Brendan	10	28	22	50	27	27	30	25
# Lesley & Anton	    11	23	26	49	27	31	24
# Naga & Pasha	        12	23	23	46	25	24
# Will & Karen	        13	30	27	57	31
# Tameka & Gorka	    14	26	29	55	28
# Melvin & Janette	    15	22	23	45

# Celebrity	Known for	Professional partner	Status
# Melvin Odoom	Television & radio broadcaster	Janette Manrara	Eliminated 1st
# on 2 October 2016
# Tameka Empson	EastEnders actress	Gorka Mrquez	Eliminated 2nd
# on 9 October 2016
# Will Young	Singer-songwriter & actor	Karen Clifton	Withdrew
# on 11 October 2016
# Naga Munchetty	BBC Breakfast newsreader & journalist	Pasha Kovalev	Eliminated 3rd
# on 16 October 2016
# Lesley Joseph	Stage & screen actress	Anton du Beke	Eliminated 4th
# on 23 October 2016
# Anastacia	Singer-songwriter	Brendan Cole
# Gorka Mrquez (Week 5)	Eliminated 5th
# on 30 October 2016
# Laura Whitmore	Television presenter	Giovanni Pernice	Eliminated 6th
# on 6 November 2016
# Daisy Lowe	Fashion model	Alja Skorjanec	Eliminated 7th
# on 13 November 2016
# Greg Rutherford	Olympic long jumper	Natalie Lowe	Eliminated 8th
# on 20 November 2016
# Ed Balls	Former Labour Party politician	Katya Jones	Eliminated 9th
# on 27 November 2016
# Judge Rinder	Criminal law barrister & television judge	Oksana Platero	Eliminated 10th
# on 4 December 2016
# Claudia Fragapane	Olympic artistic gymnast	AJ Pritchard	Eliminated 11th
# on 11 December 2016
# Louise Redknapp	Former Eternal singer & television presenter	Kevin Clifton	Runners-up
# on 17 December 2016
# Danny Mac	Former Hollyoaks actor	Oti Mabuse
# Ore Oduba	BBC Sport presenter	Joanne Clifton	Winners
# on 17 December 2016

series = 14

competitor_name_dict = {}
competitor_name_dict[1] = 'Ore Oduba'
competitor_name_dict[2] = 'Danny Mac'
competitor_name_dict[3] = 'Louise Redknapp'
competitor_name_dict[4] = 'Claudia Fragapane'
competitor_name_dict[5] = 'Judge Rinder'
competitor_name_dict[6] = 'Ed Balls'
competitor_name_dict[7] = 'Greg Rutherford'
competitor_name_dict[8] = 'Daisy Lowe'
competitor_name_dict[9] = 'Laura Whitmore'
competitor_name_dict[10] = 'Anastacia'
competitor_name_dict[11] = 'Lesley Joseph'
competitor_name_dict[12] = 'Naga Munchetty'
competitor_name_dict[13] = 'Will Young'
competitor_name_dict[14] = 'Tameka Empson'
competitor_name_dict[15] = 'Melvin Odoom'

competitor_name_by_series_dict[series] = competitor_name_dict

competitor_position_dict = {}

for i in range(15):
    competitor_position_dict[i+1] = i+1

competitor_position_by_series_dict = {}
competitor_position_by_series_dict[series] = competitor_position_dict


#use max judge score to mormalise judge scores into [0,1]
max_judge_score[14][2] = 80
max_judge_score[14][3] = 40
max_judge_score[14][4] = 40
max_judge_score[14][5] = 40
max_judge_score[14][6] = 40
max_judge_score[14][7] = 40
max_judge_score[14][8] = 40
max_judge_score[14][9] = 40
max_judge_score[14][10] = 46
max_judge_score[14][11] = 40
max_judge_score[14][12] = 80
max_judge_score[14][13] = 120

judge_score[14][2][1] = 54
judge_score[14][3][1] = 35
judge_score[14][4][1] = 39
judge_score[14][5][1] = 36
judge_score[14][6][1] = 32
judge_score[14][7][1] = 34
judge_score[14][8][1] = 35
judge_score[14][9][1] = 38
judge_score[14][10][1] = 40
judge_score[14][11][1] = 36
judge_score[14][12][1] = 77
judge_score[14][13][1] = 119

judge_score[14][2][2] = 63
judge_score[14][3][2] = 36
judge_score[14][4][2] = 36
judge_score[14][5][2] = 35
judge_score[14][6][2] = 30
judge_score[14][7][2] = 38
judge_score[14][8][2] = 38
judge_score[14][9][2] = 40
judge_score[14][10][2] = 45
judge_score[14][11][2] = 38
judge_score[14][12][2] = 76
judge_score[14][13][2] = 116

judge_score[14][2][3] = 63
judge_score[14][3][3] = 31
judge_score[14][4][3] = 33
judge_score[14][5][3] = 33
judge_score[14][6][3] = 35
judge_score[14][7][3] = 39
judge_score[14][8][3] = 37
judge_score[14][9][3] = 38
judge_score[14][10][3] = 45
judge_score[14][11][3] = 37
judge_score[14][12][3] = 73
judge_score[14][13][3] = 116

judge_score[14][2][4] = 56
judge_score[14][3][4] = 36
judge_score[14][4][4] = 30
judge_score[14][5][4] = 32
judge_score[14][6][4] = 36
judge_score[14][7][4] = 33
judge_score[14][8][4] = 36
judge_score[14][9][4] = 36
judge_score[14][10][4] = 39
judge_score[14][11][4] = 37
judge_score[14][12][4] = 73

judge_score[14][2][5] = 52
judge_score[14][3][5] = 27
judge_score[14][4][5] = 27
judge_score[14][5][5] = 29
judge_score[14][6][5] = 32
judge_score[14][7][5] = 33
judge_score[14][8][5] = 33
judge_score[14][9][5] = 33
judge_score[14][10][5] = 31
judge_score[14][11][5] = 31

judge_score[14][2][6] = 44
judge_score[14][3][6] = 24
judge_score[14][4][6] = 16
judge_score[14][5][6] = 18
judge_score[14][6][6] = 26
judge_score[14][7][6] = 27
judge_score[14][8][6] = 25
judge_score[14][9][6] = 23
judge_score[14][10][6] = 24

judge_score[14][2][7] = 53
judge_score[14][3][7] = 32
judge_score[14][4][7] = 28
judge_score[14][5][7] = 24
judge_score[14][6][7] = 26
judge_score[14][7][7] = 32
judge_score[14][8][7] = 31
judge_score[14][9][7] = 32

judge_score[14][2][8] = 62
judge_score[14][3][8] = 31
judge_score[14][4][8] = 31
judge_score[14][5][8] = 32
judge_score[14][6][8] = 33
judge_score[14][7][8] = 34
judge_score[14][8][8] = 31

judge_score[14][2][9] = 57
judge_score[14][3][9] = 30
judge_score[14][4][9] = 33
#judge_score[14][5][9] = 0 missed a week
judge_score[14][6][9] = 36
judge_score[14][7][9] = 32

judge_score[14][2][10] = 50
judge_score[14][3][10] = 27
judge_score[14][4][10] = 27
judge_score[14][5][10] = 30
judge_score[14][6][10] = 25

judge_score[14][2][10] = 50
judge_score[14][3][10] = 27
judge_score[14][4][10] = 27
judge_score[14][5][10] = 30
judge_score[14][6][10] = 25

judge_score[14][2][11] = 49
judge_score[14][3][11] = 27
judge_score[14][4][11] = 31
judge_score[14][5][11] = 24

judge_score[14][2][12] = 46
judge_score[14][3][12] = 25
judge_score[14][4][12] = 24

judge_score[14][2][13] = 57
judge_score[14][3][13] = 31
judge_score[14][4][13] = 0

judge_score[14][2][14] = 55
judge_score[14][3][14] = 28

judge_score[14][2][15] = 45

judge_score_counts[14][13] = False

#input dance-offs

competitor_in_dance_off[14][2][15] = True
competitor_in_dance_off[14][2][10] = True

competitor_in_dance_off[14][3][9] = True
competitor_in_dance_off[14][3][14] = True

competitor_in_dance_off[14][4][10] = True
competitor_in_dance_off[14][4][12] = True

competitor_in_dance_off[14][5][8] = True
competitor_in_dance_off[14][5][11] = True

competitor_in_dance_off[14][6][8] = True
competitor_in_dance_off[14][6][10] = True

competitor_in_dance_off[14][7][1] = True
competitor_in_dance_off[14][7][9] = True

competitor_in_dance_off[14][8][7] = True
competitor_in_dance_off[14][8][8] = True

competitor_in_dance_off[14][9][4] = True
competitor_in_dance_off[14][9][7] = True

competitor_in_dance_off[14][10][5] = True
competitor_in_dance_off[14][10][6] = True

competitor_in_dance_off[14][11][1] = True
competitor_in_dance_off[14][11][5] = True

competitor_in_dance_off[14][12][2] = True
competitor_in_dance_off[14][12][4] = True

competitor_in_dance_off[14][13][2] = True
competitor_in_dance_off[14][13][3] = True

for k, v in competitor_name_dict.items():
    print k, v

print judge_score.shape



#********************************************************************************************************************
#
# Build derived data structures

final_round_in_series = {}
series_list = []

valid_series_round, valid_series_competitor, series_round_competitor_dict  = create_competitor_lists(judge_score)
series_competitor_dict = {}
series_round_dict = {}

print '** Rounds in each series'

for series, round_dict in series_round_competitor_dict.items():
    rounds_in_series_list = []
    for round, competitor_list in round_dict.items():
        rounds_in_series_list.append(round)
    series_round_dict[series] = sorted(set(rounds_in_series_list))

for i, v in series_round_dict.items():
    print i, v


print '** Competitors'

for series, round_dict in series_round_competitor_dict.items():
    competitors_in_series_list = []
    series_list.append(series)
    final_round = 0
    for round, competitor_list in round_dict.items():
        if round > final_round:
            final_round = round
        print series, round, competitor_list
        for competitor in competitor_list:
            competitors_in_series_list.append(competitor)
    series_competitor_dict[series] = sorted(set(competitors_in_series_list))
    final_round_in_series[series] = final_round

print '** Dance off competitors'

series_round_competitor_dance_off_dict = create_danceoff_dict(competitor_in_dance_off)

for series, round_dict in series_round_competitor_dance_off_dict.items():
    for round, competitor_list in round_dict.items():
        print series, round, competitor_list

print '** Final round in series'

for series, final_round in final_round_in_series.items():
    print series, final_round

print '** Normalised judge scores'

judge_score_normalised = np.zeros((num_series, max_rounds, max_competitors))

for series in series_list:
    for round in range(max_rounds):
        for competitor in range(max_competitors):
            judge_score_normalised[series, round, competitor] = max_judge_score_norm * judge_score[series, round, competitor] / max_judge_score[series, round]
            print series, round, competitor, judge_score[series, round, competitor], judge_score_normalised[series, round, competitor]

print '** Number of dancers in dance-off'

#usually two but in finals can be 1,2 or 3

num_dancers_dance_off_round = np.zeros((num_series, max_rounds), dtype=int)

for series in series_list:
    for round in range(max_rounds):
        for competitor in range(max_competitors):
            if competitor_in_dance_off[series][round][competitor]:
                num_dancers_dance_off_round[series][round] += 1




#********************************************************************************************************************
#
# Compute judge's ranked scores

for series, round_dict in series_round_competitor_dict.items():
    for round, competitor_list in round_dict.items():
        judge_norm_scores = judge_score_normalised[series, round, :]
        judge_points[series, round, :] = convert_score_to_points(judge_norm_scores)

# for series, round_dict in series_round_competitor_dict.items():
#     for round, competitor_list in round_dict.items():
#         for competitor in competitor_list:
#             print series, round, competitor, judge_score[series, round, competitor], "%.2f" % judge_score_normalised[series, round, competitor], judge_points[series, round, competitor]


# ********************************************************************************************************************
#
# Initialise competitor popularity

print 'Initialise competitor popularity'

# initialise chain 0 based on simple popularity measure based on dance-off vs. judge scores

for series in series_list:
    print 'series', series

    # initialise competitots in this series to zero
    competitor_series_popularity_iterations[0, series, series_competitor_dict[series], 0] = 0

    for round in series_round_dict[series]:
        print 'round', round

        #competitors in dance off
        competitors_in_dance_off = series_round_competitor_dance_off_dict[series][round]
        print 'competitors_in_dance_off', competitors_in_dance_off

        #bottom 2 competitors on judge score
        ranked_combined_scores = np.argsort(judge_score[series, round, :])
        judges_bottom_two = ranked_combined_scores[:2]
        print 'judges_bottom_two', judges_bottom_two

        popular_competitors = np.setdiff1d(judges_bottom_two, competitors_in_dance_off)
        unpopular_competitors = np.setdiff1d(competitors_in_dance_off, judges_bottom_two)

        print 'popular_competitors', popular_competitors
        print 'unpopular_competitors', unpopular_competitors

        competitor_series_popularity_iterations[0, series, popular_competitors, 0] = 1.0
        competitor_series_popularity_iterations[0, series, unpopular_competitors, 0] = -1.0

    print competitor_series_popularity_iterations[0, series, :, 0]

    competitor_series_popularity_iterations[0, series, :, 0] = normalise_vector_std_normal(competitor_series_popularity_iterations[0, series, :, 0])

    print competitor_series_popularity_iterations[0, series, :, 0]

    # initialise the other chains with simply random starts

    for chain in range(1, num_chains):

        print 'chain', chain

        for competitor in series_competitor_dict[series]:

            competitor_series_popularity_iterations[chain, series, competitor, 0] = np.random.rand()

        print competitor_series_popularity_iterations[chain, series, :, 0]

        competitor_series_popularity_iterations[chain, series, :, 0] = normalise_vector_std_normal(competitor_series_popularity_iterations[chain, series, :, 0])

        print competitor_series_popularity_iterations[chain, series, :, 0]



# ********************************************************************************************************************
#
# Initialise variable parameters

print 'Initialise variable parameters'

for chain in range(num_chains):

    # judge_score_weight = variable_params[0]
    variable_parameters_iterations[chain][0][0] = np.random.rand()

    # epsilon_counts = variable_params[1]
    variable_parameters_iterations[chain][1][0] = 1

    # noise_std_dev = variable_params[2]
    variable_parameters_iterations[chain][2][0] = 0.2

    # std dev of the normal distribution from which the next step is taken
    variable_parameters_iterations[chain][3][0] = 1.0


#********************************************************************************************************************
#
# Main program

while not converged:

    print 'block_number', block_number

    for block_iteration in range(0, block_size): #initial settings are in 0

        print 'block_iteration', block_iteration

        iteration = block_iteration + (block_number * block_size)

        for chain in range(num_chains):

            #print 'chain', chain

            variable_parameters_live = variable_parameters_iterations[chain, :, iteration]

            #print 'iteration', iteration

            for series in series_list:

                #print 'series', series

                #initialise competitor popularity and variable parameters by copying values from the
                # previous iteration into this iteration
                competitor_popularity_live = competitor_series_popularity_iterations[chain, series, :, iteration]

                comp_prob_norm_std = variable_parameters_live[3]

                # compute series_log_like_live for live values of competitor popularity and variable parameters
                series_log_like_live = series_log_like(series, competitor_popularity_live, variable_parameters_live)

                max_series_log_like_live = series_log_like_live

                for competitor in range(max_competitors):
                    if valid_series_competitor[series, competitor]:

                        competitor_popularity_live_variant = np.copy(competitor_popularity_live)
                        # make a step from current value for active competitor

                        competitor_popularity_live_variant[competitor] = np.random.normal(competitor_popularity_live[competitor], comp_prob_norm_std)

                        # normalise the vector
                        competitor_popularity_live_variant = normalise_vector_std_normal(competitor_popularity_live_variant)

                        # compute series log-likelihood for this variant
                        series_log_like_variant = series_log_like(series, competitor_popularity_live_variant, variable_parameters_live)

                        # sample existing OR variant based on probabilities
                        base_prob_non_norm = np.exp(series_log_like_live) + underflow_avoidance
                        variant_prob_non_norm = np.exp(series_log_like_variant) + underflow_avoidance
                        total_prob = base_prob_non_norm + variant_prob_non_norm
                        base_prob_norm = base_prob_non_norm / total_prob

                        rand_value = np.random.rand()

                        if rand_value > base_prob_norm:
                            select_variant = True
                            #then set live to be equal to the variant
                            competitor_popularity_live = np.copy(competitor_popularity_live_variant)
                            series_log_like_live = series_log_like_variant

                        else:
                            select_variant = False
                            #competitor_popularity_live = competitor_popularity_live_base.copy
                            #competitor_popularity_live is unchanged

                        if series_log_like_live > max_series_log_like_live:
                            max_series_log_like_live = series_log_like_live

                # now at this point it should have looped through all the competitors and we have a revised
                # competitor_popularity_live vector for this series
                # using this revised vector we do the same process for the variable parameters

                competitor_series_popularity_iterations[chain, series, :, iteration+1] = competitor_popularity_live

                series_log_like_live_array[chain, series, iteration+1] = max_series_log_like_live

            #write the set of competitor popularities for all series for this iteration
            competitor_series_popularity = competitor_series_popularity_iterations[chain, :, :, iteration+1]

            #set the live value
            all_series_log_like_live = all_series_log_likelihood(competitor_series_popularity, variable_parameters_live)

            ############################################################################################
            #
            # 0 judge_score_weight

            beta_width_factor = 10.0
            judge_score_weight_min = 0.0
            judge_score_weight_max = 1.0
            judge_score_weight_span = judge_score_weight_max - judge_score_weight_min

            variable_parameter_live = variable_parameters_live[0]

            mode = 1.0 * (variable_parameter_live - judge_score_weight_min) / judge_score_weight_span
            a = (mode * (1.0 * beta_width_factor - 2.0)) + 1.0
            b = beta_width_factor - a
            mode_new = np.random.beta(a, b)

            variable_parameter_variant = (mode_new * judge_score_weight_span) + judge_score_weight_min

            variable_parameters_variant = np.copy(variable_parameters_live)
            variable_parameters_variant[0] = variable_parameter_variant

            all_series_log_like_variant = all_series_log_likelihood(competitor_series_popularity, variable_parameters_variant)

            base_prob_non_norm = np.exp(all_series_log_like_live) + underflow_avoidance
            variant_prob_non_norm = np.exp(all_series_log_like_variant) + underflow_avoidance
            total_prob = base_prob_non_norm + variant_prob_non_norm
            base_prob_norm = base_prob_non_norm / total_prob

            rand_value = np.random.rand()

            if rand_value > base_prob_norm:
                select_variant = True
                # then set live to be equal to the variant
                variable_parameters_live = np.copy(variable_parameters_variant)
                all_series_log_like_live = all_series_log_like_variant

            else:
                select_variant = False
                # then keep current live values

            #print 'judge_score_weight', variable_parameter_live

            ############################################################################################
            #
            # 1 epsilon_counts

            # leave fixed for now

            ############################################################################################
            #
            # 2 noise_std_dev

            # leave fixed for now


            ############################################################################################
            #
            # 3 step_std_dev

            # just do with a learning rate

            step_std_dev_initial_value = variable_parameters_iterations[chain][3][0]
            step_std_dev_final_value = 0.3
            step_std_dev_change_per_iter = (step_std_dev_final_value - step_std_dev_initial_value) / (1.0 * max_iters)

            variable_parameters_live[3] = step_std_dev_initial_value + (iteration * step_std_dev_change_per_iter)

            #print 'step_std_dev', variable_parameters_live[3]

            ############################################################################################
            #
            # copy the updated variable parameter values into the iterations array
            variable_parameters_iterations[chain, :, iteration+1] = variable_parameters_live

    # check convergence for block

    print '*** Convergence'

    start_iter = block_number * block_size
    end_iter = (block_number+1) * block_size

    max_conv = 0

    for series in series_list:

        print 'series', series

        conv_stat_comp = gelman_rubin_diagnostic(competitor_series_popularity_iterations[:, series, :, start_iter:end_iter])
        print 'conv_stat_comp', conv_stat_comp
        print 'mean', np.nanmean(conv_stat_comp)
        conv_stat_comp_max = np.nanmax(conv_stat_comp)
        print 'max', conv_stat_comp_max

        if conv_stat_comp_max > max_conv:
            max_conv = conv_stat_comp_max

    # only interested in parameter 0
    conv_stat_var_param = gelman_rubin_diagnostic(variable_parameters_iterations[:, :, start_iter:end_iter])[0]

    print 'conv_stat_var_param', conv_stat_var_param

    if conv_stat_var_param > max_conv:
        max_conv = conv_stat_var_param

    print 'max_conv', max_conv

    max_conv_block[block_number] = max_conv

    if block_number+1 > burn_in_blocks:
        if max_conv<convergence_target:
            converged = True  # to exit from while loop

    block_number += 1

    if block_number == max_num_blocks:
        converged = True  # to exit from while loop

    number_of_blocks = block_number

    print 'number_of_blocks', number_of_blocks


# ********************************************************************************************************************
#
# Quick and dirty display distributions of popularity and compute expected values

for series in series_list:

    competitor_popularity_list_dict = {}
    competitor_popularity_dict = {}

    print 'Series', series

    competitor_name_dict = competitor_name_by_series_dict[series]

    for competitor, competitor_name in competitor_name_dict.items():

        print competitor, competitor_name

        for block in range(number_of_blocks):

            # slice_start = (1.0 - (slice_prop * num_slices) + (slice * slice_prop)) * num_iters
            block_start_iter = int(block * block_size)
            block_end_iter = int((block + 1) * block_size)

            #print slice_start, slice_end

            competitor_series_popularity_iterations_block = competitor_series_popularity_iterations[:, series, competitor,
                                                            block_start_iter:block_end_iter]

            competitor_popularity_block_mean = np.mean(competitor_series_popularity_iterations_block)

            print block_start_iter, 'to', block_end_iter, "%.3f" % competitor_popularity_block_mean

    for variable_param in range(num_variable_parameters):

        print 'variable_param', variable_param

        for block in range(number_of_blocks):
            # slice_start = (1.0 - (slice_prop * num_slices) + (slice * slice_prop)) * num_iters
            block_start_iter = int(block * block_size)
            block_end_iter = int((block + 1) * block_size)

            variable_parameters_block = variable_parameters_iterations[:, variable_param, block_start_iter:block_end_iter]

            variable_parameters_block_mean = np.mean(variable_parameters_block)

            print block_start_iter, 'to', block_end_iter, "%.3f" % variable_parameters_block_mean



# ********************************************************************************************************************
#
# Output results to files

filepath = '/Users/David/Documents/Documents - iMac/Education and Qualifications/Sussex PhD/UoS_PhD/H_Strictly_Bayesian/data/results/'

# ********************************************************************************************************************
#
# competitor popularity

#set up dictionaries

results_comp_pop = OrderedDict()

results_comp_pop['series'] = {}
results_comp_pop['block'] = {}
results_comp_pop['iter_start'] = {}
results_comp_pop['iter_end'] = {}

for competitor in range(max_competitors):
    results_comp_pop[competitor] = {}

# put results into dictionaries

row=0

for series in series_list:
    for block in range(number_of_blocks):
        block_start_iter = int(block * max_iters / number_of_blocks)
        block_end_iter = int((block + 1) * max_iters / number_of_blocks)

        row +=1

        results_comp_pop['series'][row] = series
        results_comp_pop['block'][row] = block
        results_comp_pop['iter_start'][row] = block_start_iter
        results_comp_pop['iter_end'][row] = block_end_iter

        for competitor in range(max_competitors):

            results_comp_pop[competitor][row] = np.mean(competitor_series_popularity_iterations[:, series, competitor, block_start_iter:block_end_iter])

# ********************************************************************************************************************
#
# non competitor-series values

results_non_series = OrderedDict()

results_non_series['block'] = {}
results_non_series['iter_start'] = {}
results_non_series['iter_end'] = {}
results_non_series['judge_score_weight'] = {}
results_non_series['epsilon_counts'] = {}
results_non_series['noise_std_dev'] = {}
results_non_series['step_std_dev'] = {}
results_non_series['max_conv'] = {}

for block in range(number_of_blocks):
    block_start_iter = int(block * max_iters / number_of_blocks)
    block_end_iter = int((block + 1) * max_iters / number_of_blocks)

    results_non_series['block'][block] = block
    results_non_series['iter_start'][block] = block_start_iter
    results_non_series['iter_end'][block] = block_end_iter
    results_non_series['judge_score_weight'][block] = np.mean(variable_parameters_iterations[:, 0, block_start_iter:block_end_iter])
    results_non_series['epsilon_counts'][block] = np.mean(variable_parameters_iterations[:, 1, block_start_iter:block_end_iter])
    results_non_series['noise_std_dev'][block] = np.mean(variable_parameters_iterations[:, 2, block_start_iter:block_end_iter])
    results_non_series['step_std_dev'][block] = np.mean(variable_parameters_iterations[:, 3, block_start_iter:block_end_iter])
    results_non_series['max_conv'][block] = max_conv_block[block]

#################################################################################################
#
# Convert dictionaries to df and output

df_comp_pop = pd.DataFrame.from_dict(results_comp_pop)
filename = 'competitor_popularity_' + timestamp + '.csv'
df_comp_pop.to_csv(filepath + filename)

df_competitors = pd.DataFrame.from_dict(competitor_name_by_series_dict)
filename = 'competitor_names_' + timestamp + '.csv'
df_competitors.to_csv(filepath + filename)

df_results_non_series = pd.DataFrame.from_dict(results_non_series)
filename = 'results_non_series_' + timestamp + '.csv'
df_results_non_series.to_csv(filepath + filename)


print df_comp_pop
print df_competitors
print df_results_non_series

print df_results_non_series.mean()

