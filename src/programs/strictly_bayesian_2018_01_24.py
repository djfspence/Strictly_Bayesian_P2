__author__ = 'David'

#********************************************************************************************************************
#
# Imports

#import pymc as pm
import numpy as np
import sys
import random
import pandas as pd



#********************************************************************************************************************
#
# To Do

# Change from uniform priors to beta priors with mode set to current value and width of beta gradually reducing as the
# number of iterations increases. i.e. wide exploration initially but typically finer later.

# Track the proportion of keeping current vs. choosing alternative


# Weve also seen the curse of the middle of the leaderboard in operation like never before. When couples perform
# solidly and finish halfway up the standings, viewers frequently forget to vote for them. Theyve neither been blown
# away by their performance, nor feel compelled to save their skin.

# Add some sort of "save their skin" factor that boosts the scores of competitors at risk?
# Lower than normal judge score?
# Survived a dance-off? This one might be better...

#********************************************************************************************************************
#
# Fixed parameters

num_series = 15
max_rounds = 20
max_competitors = 20
casting_vote_multiplier = 1.001
num_variable_parameters = 4
underflow_avoidance = 1e-100   #add to the base and variant probbailities to avoid inf / zero

num_iters = 100000
burn_in = 1

#beta distribution (maintains uniform sampling until end of burn-in)
shape_initial = 2.5
shape_final = 10
beta_change_iter_prop = 0.5

#fixed_parameters

num_prob_iters = 4 #number of iterations to compute probability of dance-off
#comp_prob_norm_std = 1.0 # normal distribution std dev from which next competitor popularity is sampled
max_judge_score_norm = 40

#********************************************************************************************************************
#
# Variable parameters

# the main one!
# popularity of each competitor in the range [0,1]

competitor_series_popularity_current = np.zeros((num_series, max_competitors))
competitor_series_popularity_current[:] = np.nan

competitor_series_popularity_alternate = np.zeros((num_series, max_competitors))
competitor_series_popularity_alternate[:] = np.nan

competitor_series_popularity_iterations = np.zeros((num_series, max_competitors, num_iters))
competitor_series_popularity_iterations[:] = np.nan

variable_parameters_current = np.zeros((num_variable_parameters))
variable_parameters_current[:] = np.nan

variable_parameters_alternate = np.zeros((num_variable_parameters))
variable_parameters_alternate[:] = np.nan

variable_parameters_iterations = np.zeros((num_variable_parameters, num_iters))
variable_parameters_iterations[:] = np.nan

log_likelihood_series_current = np.zeros(num_series)  #need to initialise this on first run
log_likelihood_series_current[:] = np.nan

log_likelihood_series_iterations = np.zeros((num_series, num_iters))
log_likelihood_series_iterations[:] = np.nan

series_log_like_live_array = np.zeros((num_series, num_iters))
series_log_like_live_array[:] = np.nan


# just the sum of the above
log_likelihood_total_current = 0   #need to initialise this on first run
log_likelihood_total_iterations = np.zeros(num_iters)

#parameters
# 0 proportion of judges' score in audience score [0,1]
# 1 value added to iteration for probability of dance-off to ensure probability is not 0 or 1
# 2 variance of normal distribution from which audience score 'noise' is selected


#********************************************************************************************************************
#
# Functions

def round_log_likelihood(series, round, competitor_popularity, variable_params):

    '''compute log likelihood of dance-offs for a round given the competitor_popularity and variable-params'''

    #print 'competitor_popularity', competitor_popularity
    competitors_in_round = np.where(~np.isnan(competitor_popularity))[0]

    num_competitors = competitor_popularity.shape[0]
    audience_score_plus_noise = np.zeros((num_competitors, num_prob_iters))
    #competitors_in_round = np.argwhere(np.isfinite(competitor_popularity))

    judge_score_weight = variable_params[0]
    epsilon_counts = int(variable_params[1])
    noise_std_dev = variable_params[2]
    judge_score_round_norm = np.zeros(num_competitors)
    audience_score_round = np.zeros(num_competitors)
    audience_score_round[:] = np.nan
    log_likelihood_value = np.zeros(num_competitors)
    dance_off_count = np.zeros(num_competitors, dtype=int)
    dance_off_prop = np.zeros(num_competitors)

    #print 'competitors_in_round', competitors_in_round

    #normalise the judge score so that it is in line with competitor popularity so judge score weight works ok
    judge_score_round_norm = normalise_vector_std_normal(judge_score_normalised[series, round, :])

    # for competitor in competitors_in_round:
    #     #judge_score_round_norm[competitor] = judge_score_normalised[series, round, competitor]
    #     audience_score_round[competitor] = ((1.0 - judge_score_weight) * competitor_popularity[competitor]) + (judge_score_weight * judge_score_round_norm[competitor])

    # audience_score_round[competitors_in_round] = ((1.0 - judge_score_weight) * competitor_popularity[competitors_in_round]) + (
    # judge_score_weight * judge_score_round_norm[competitors_in_round])
    #
    # audience_score_round = normalise_vector_std_normal(audience_score_round)

   # print 'audience_score_round', audience_score_round

    # audience_score_round is the combination of competitor popularity plus judge's score

    for i in range(num_prob_iters):

        #take the competitor probabilities and add noise to them

        #generate noise vector
        noise_vector = np.random.normal(0.0, noise_std_dev, num_competitors)

        competitor_popularity_plus_noise = normalise_vector_std_normal(competitor_popularity + noise_vector)

        #print 'competitor_popularity', competitor_popularity
        #print 'noise_vector', noise_vector
        #print 'competitor_popularity_plus_noise', competitor_popularity_plus_noise

        audience_score_round[competitors_in_round] = ((1.0 - judge_score_weight) * competitor_popularity_plus_noise[
            competitors_in_round]) + (judge_score_weight * judge_score_round_norm[competitors_in_round])

        audience_score_round = normalise_vector_std_normal(audience_score_round)

        audience_points = convert_score_to_points(audience_score_round)

        #combine judge score and audience score
        combined_score = judge_points[series, round, :] + audience_points

        #print 'audience_points', audience_points
        #print 'judge_points[series, round, :]', judge_points[series, round, :]
        #print 'combined_score', combined_score

        ranked_combined_scores = np.argsort(combined_score)

        #print 'ranked_combined_scores', ranked_combined_scores

        dance_off_competitors = ranked_combined_scores[:2]

        #print 'dance_off_competitors', dance_off_competitors

        dance_off_count[ranked_combined_scores[0]] += 1
        dance_off_count[ranked_combined_scores[1]] += 1

        #print 'C dance_off_count', dance_off_count

        #print 'iteration', i



    #compute probabilities

    #print 'D dance_off_count', dance_off_count

    #print 'epsilon_counts', epsilon_counts

    dance_off_count += epsilon_counts

    #print 'A dance_off_count', dance_off_count

    dance_off_prop = dance_off_count / (1.0 * (num_prob_iters + (2 * epsilon_counts)))

    dance_off_prop_cir = dance_off_prop[competitors_in_round]
    comp_in_danceoff = competitor_in_dance_off[series, round, competitors_in_round]

    #print 'B dance_off_prop_cir', dance_off_prop_cir

    # print 'comp_in_danceoff', comp_in_danceoff
    #
    # print dance_off_prop_cir[comp_in_danceoff == True]
    # print dance_off_prop_cir[comp_in_danceoff == False]
    # print 1.0 - dance_off_prop_cir[comp_in_danceoff == False]
    #
    # print np.log(dance_off_prop_cir[comp_in_danceoff == True]), np.sum(np.log(dance_off_prop_cir[comp_in_danceoff == True]))
    # print np.log(1.0 - dance_off_prop_cir[comp_in_danceoff == False]), np.sum(np.log(1.0 - dance_off_prop_cir[comp_in_danceoff == False]))

    log_prob = np.sum(np.log(dance_off_prop_cir[comp_in_danceoff == True])) + np.sum(np.log(1.0 - dance_off_prop_cir[comp_in_danceoff == False]))

    #print 'log_prob', log_prob

    return log_prob

def series_log_like(series, competitor_popularity, variable_parameters):

    '''Compute log_likelihood of observed dance-offs for series'''

    series_log_likelihood = 0

    for round in range(max_rounds):
        if valid_series_round[series, round]:

            #print 'valid series round', series, round

            # then this was a scored round in that series
            round_log_like = round_log_likelihood(series, round, competitor_popularity,
                                                  variable_parameters)
            series_log_likelihood += round_log_like

        #else:
            #print 'invalid series round', series, round

    #print 'series_log_likelihood', series_log_likelihood

    return series_log_likelihood


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

    #print 'raw_scores', raw_scores
    #print 'point_scores', point_scores

    return point_scores



def normalise_vector_std_normal(vector):

    '''Takes an input vector and normalises it to zero mean and unit std_dev'''

    #to be safe convert any zeros to nan

    #vector[vector==0] = np.nan

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

    #print series_set

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


def sample_from_log_likelihoods(current_ll, alternate_ll):

    current_likelihood = np.exp(current_ll)
    alternate_likelihood = np.exp(alternate_ll)

    total_likelihood = current_likelihood + alternate_likelihood

    current_likelihood_norm = current_likelihood / total_likelihood
    alternate_likelihood_norm = alternate_likelihood / total_likelihood

    rand_value = np.random.rand()

    keep_current = False
    if rand_value < current_likelihood_norm:
        keep_current = True

    return keep_current


def sample_beta(mode, shape):

    '''return a sample from a beta distribution of given mode and shape'''

    m = 1.0 * mode
    s = 1.0 * shape

    a = m * (s - 2.0) + 1.0
    b = s - 1.0 - (m * (s - 2.0))

    return np.random.beta(a, b)




#********************************************************************************************************************
#
# Set up data structures

# competitors by series

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

# competitor_in_dance_off = np.zeros((num_series, max_rounds, max_competitors), dtype=int)
# competitor_in_dance_off[:] = np.nan

competitor_in_dance_off = np.full((num_series, max_rounds, max_competitors), False, dtype=bool)


# =1 if this was an elimination round in the series
#valid_series_round = np.zeros((num_series, max_rounds), dtype=int)
#valid_series_competitor = np.zeros((num_series, max_competitors), dtype=int)
#num_competitors_in_round = np.zeros((num_series, max_rounds))


competitor_name_dict = {}

# need to do this by import from csv ideally

# data for series 14

series = 14

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

competitor_name_by_series_dict = {}
competitor_name_by_series_dict[series] = competitor_name_dict

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
judge_score[14][5][9] = 0
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
# Parameters





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

#OK so this now gives me access to lists that I can iterate over

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



#********************************************************************************************************************
#
# Compute judge's ranked scores

for series, round_dict in series_round_competitor_dict.items():
    for round, competitor_list in round_dict.items():
        judge_norm_scores = judge_score_normalised[series, round, :]
        judge_points[series, round, :] = convert_score_to_points(judge_norm_scores)

        #print 'series', series, 'round', round, judge_points[series, round, :]


#test

for series, round_dict in series_round_competitor_dict.items():
    for round, competitor_list in round_dict.items():
        for competitor in competitor_list:
            print series, round, competitor, judge_score[series, round, competitor], "%.2f" % judge_score_normalised[series, round, competitor], judge_points[series, round, competitor]



# ********************************************************************************************************************
#
# Initialise competitor popularity

print 'Initialise competitor popularity'

# for series, competitor_list in series_competitor_dict.items():
#     for competitor in competitor_list:
#         competitor_series_popularity_iterations[series, competitor, 0] = np.random.normal(0.0, 1.0)
#         print series, competitor, competitor_series_popularity_iterations[series, competitor, 0]
#
#     #normalise the competitor popularity values to zero mean and unit std dev
#
#     print competitor_series_popularity_iterations[series, :, 0], np.sum(competitor_series_popularity_iterations[series, :, 0])
#
#     competitor_series_popularity_iterations[series, :, 0] = normalise_vector_std_normal(competitor_series_popularity_iterations[series, :, 0])
#
#     print competitor_series_popularity_iterations[series, :, 0], np.sum(competitor_series_popularity_iterations[series, :, 0])

    # do a smarter version based on dance off vs. judge scores
    # all competitors start at 0
    # loop through the rounds
    # competitor in deance off but not in bottom 2 of judge's scores = -1
    # competitor not in dance off but in bottom 2 of judge's scores = +1
    # then normalise

for series in series_list:
    print 'series', series

    # initialise competitots in this series to zero
    competitor_series_popularity_iterations[series, series_competitor_dict[series], 0] = 0

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

        competitor_series_popularity_iterations[series, popular_competitors, 0] = 1.0
        competitor_series_popularity_iterations[series, unpopular_competitors, 0] = -1.0

    print competitor_series_popularity_iterations[series, :, 0]

    competitor_series_popularity_iterations[series, :, 0] = normalise_vector_std_normal(competitor_series_popularity_iterations[series, :, 0])

    print competitor_series_popularity_iterations[series, :, 0]


# ********************************************************************************************************************
#
# Initialise variable parameters

print 'Initialise variable parameters'

# judge_score_weight = variable_params[0] - range[0,1]
variable_parameters_iterations[0][0] = 0.3

# epsilon_counts = variable_params[1] - say 1 count
variable_parameters_iterations[1][0] = 1

# noise_std_dev = variable_params[2] probably much less than the unit std dev
variable_parameters_iterations[2][0] = 1.0

# std dev of the normal distribution from which the next step is taken
variable_parameters_iterations[3][0] = 1.0


#********************************************************************************************************************
#
# Main program

variable_parameters_live = variable_parameters_iterations[:, 0]

for series in series_list:

    print 'series', series

    for iteration in range(1, num_iters): #initial settings are in 0

        print 'iteration', iteration

        #initialise competitor popularity and variable parameters by copying values from the
        # previous iteration into this iteration

        competitor_popularity_live = competitor_series_popularity_iterations[series, :, iteration - 1]

        #print 'competitor_popularity_live', competitor_popularity_live
        #print 'variable_parameters_live', variable_parameters_live

        #consider putting a learning rate in here linked to iterations
        #initially set this quite high then narrow it down?
        comp_prob_norm_std = variable_parameters_live[3]

        # compute series_log_like_live for live values of competitor popularity and variable parameters

        series_log_like_live = series_log_like(series, competitor_popularity_live, variable_parameters_live)

        max_series_log_like_live = series_log_like_live

        for competitor in range(max_competitors):
            if valid_series_competitor[series, competitor]:

                competitor_popularity_live_variant = np.copy(competitor_popularity_live)
                # make a step from current value for active competitor
                #print 'competitor_popularity_live[competitor]', competitor_popularity_live[competitor]
                #print 'comp_prob_norm_std', comp_prob_norm_std

                competitor_popularity_live_variant[competitor] = np.random.normal(competitor_popularity_live[competitor], comp_prob_norm_std)
                #print competitor_popularity_live_variant[competitor]

                # normalise the vector

                competitor_popularity_live_variant = normalise_vector_std_normal(competitor_popularity_live_variant)

                # compute series log-likelihood for this variant

                series_log_like_variant = series_log_like(series, competitor_popularity_live_variant, variable_parameters_live)

                # sample existing OR variant based on probabilities

                base_prob_non_norm = np.exp(series_log_like_live) + underflow_avoidance
                variant_prob_non_norm = np.exp(series_log_like_variant) + underflow_avoidance
                total_prob = base_prob_non_norm + variant_prob_non_norm
                base_prob_norm = base_prob_non_norm / total_prob

                #print 'base_prob_non_norm', base_prob_non_norm
                #print 'variant_prob_non_norm', variant_prob_non_norm
                #print 'total_prob', total_prob
                #print 'base_prob_norm', base_prob_norm

                rand_value = np.random.rand()
                #print 'rand_value', rand_value

                if rand_value > base_prob_norm:
                    select_variant = True
                    #then set live to be equal to the variant
                    competitor_popularity_live = np.copy(competitor_popularity_live_variant)
                    series_log_like_live = series_log_like_variant

                else:
                    select_variant = False
                    #competitor_popularity_live = competitor_popularity_live_base.copy
                    #competitor_popularity_live is unchanged

                #print 'select_variant', select_variant

                if series_log_like_live > max_series_log_like_live:
                    max_series_log_like_live = series_log_like_live

        # now at this point it should have looped through all the competitors and we have a revised
        # competitor_popularity_live vector for this series
        # using this revised vector we do the same process for the variable parameters

            #loop over variable parameters and pick most likely
                #select alternative value for parameter
                #compute probability of both values
                #sample from those two probabilities and pick new value
                #update parameter value

        # write iteration results to file

        competitor_series_popularity_iterations[series, :, iteration] = competitor_popularity_live

        #print 'max_series_log_like_live', max_series_log_like_live

        series_log_like_live_array[series, iteration] = max_series_log_like_live


#print 'variable parameters'

#for variable_parameter in range(num_variable_parameters):
    #print variable_parameter, variable_parameters_live[variable_parameter]

    # judge_score_weight = variable_params[0] - range[0,1]
    # epsilon = variable_params[1]
    # noise_std_dev = np.sqrt(variable_params[2])

    # the log likelihood has just been computed for current settings

    # ***** come back to this later, for now leave these as fixed values ******



#do the same process for the dance_score_weight which is series independent


# ********************************************************************************************************************
#
# Display distributions of popularity and compute expected values


series = 14

competitor_popularity_iterations = np.zeros((max_competitors, num_iters))
competitor_popularity_iterations[:] = np.nan

competitor_popularity_iterations = competitor_series_popularity_iterations[series, :, :]

for competitor in range(max_competitors):
    print 'competitor', competitor
    for iteration in range(num_iters):
        print iteration, competitor_popularity_iterations[competitor, iteration]

for iteration in range(num_iters):
    print series_log_like_live_array[series, iteration]


print competitor_popularity_iterations

print series_log_like_live_array




# ********************************************************************************************************************
#
# Output results to file

filepath = '/Users/David/Documents/Documents - iMac/Education and Qualifications/Sussex PhD/UoS_PhD/H_Strictly_Bayesian/data/'

#series competitor names

filename = filepath + 'series_competitors.csv'
df_series_competitors = pd.DataFrame.from_dict(competitor_name_by_series_dict)
df_series_competitors.to_csv(filename)

print df_series_competitors

#competitor popularity

filename = filepath + 'competitor_popularity_iterations.csv'
df_competitor_popularity_iterations = pd.DataFrame(competitor_popularity_iterations)
df_competitor_popularity_iterations.to_csv(filename)

print df_competitor_popularity_iterations

#likelihood

filename = filepath + 'series_likelihood.csv'
df_series_likelihood = pd.DataFrame(series_log_like_live_array)
df_series_likelihood.to_csv(filename)

print df_series_likelihood




