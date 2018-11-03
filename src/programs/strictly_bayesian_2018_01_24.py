__author__ = 'David'

#********************************************************************************************************************
#
# Imports

#import pymc as pm
import numpy as np
import sys
import random



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

num_iters = 5000
burn_in = 1000

#beta distribution (maintains uniform sampling until end of burn-in)
shape_initial = 2.5
shape_final = 10
beta_change_iter_prop = 0.5

#fixed_parameters

num_prob_iters = 10 #number of iterations to compute probability of dance-off
#comp_prob_norm_std = 1.0 # normal distribution std dev from which next competitor popularity is sampled
max_judge_score_norm = 40

#********************************************************************************************************************
#
# Variable parameters

# the main one!
# popularity of each competitor in the range [0,1]

competitor_series_popularity_current = np.zeros((num_series, max_competitors))
competitor_series_popularity_alternate = np.zeros((num_series, max_competitors))
competitor_series_popularity_iterations = np.zeros((num_series, max_competitors, num_iters))

variable_parameters_current = np.zeros((num_variable_parameters))
variable_parameters_alternate = np.zeros((num_variable_parameters))
variable_parameters_iterations = np.zeros((num_variable_parameters, num_iters))

log_likelihood_series_current = np.zeros(num_series)  #need to initialise this on first run
log_likelihood_series_iterations = np.zeros(num_series, num_iters)

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

    num_competitors = competitor_popularity.shape[0]
    audience_score_plus_noise = np.zeros((num_competitors, num_prob_iters))
    competitors_in_round = np.nonzero(competitor_popularity)[0]
    judge_score_weight = variable_params[0]
    epsilon = variable_params[1]
    noise_std_dev = np.sqrt(variable_params[2])
    judge_score_round_norm = np.zeros(num_competitors)
    audience_score_round = np.zeros(num_competitors)
    dance_off_array = np.zeros((num_competitors, num_prob_iters), dtype=int)
    log_likelihood_value = np.zeros(num_competitors)
    dance_off_count = np.zeros(num_competitors, dtype=int)
    dance_off_prop = np.zeros(num_competitors)

    print 'competitors_in_round', competitors_in_round

    for competitor in competitors_in_round:
        judge_score_round_norm[competitor] = judge_score_normalised[series, round, competitor]
        audience_score_round[competitor] = ((1.0 - judge_score_weight) * competitor_popularity[competitor]) + (judge_score_weight * judge_score_round_norm[competitor])

    for i in range(num_prob_iters):

        # add normally distributed noise to the raw audience score
        for competitor in competitors_in_round:

            print 'competitor', competitor
            print 'audience_score_round[competitor]', audience_score_round[competitor]

            if audience_score_round[competitor] <> 0:

                print 'noise_std_dev', noise_std_dev

                noise = np.random.normal(0.0, noise_std_dev)
                print 'noise', noise

                audience_score_plus_noise[competitor, i] = audience_score_round[competitor] + noise

        audience_score_ranked = convert_score_to_points(audience_score_plus_noise)

        #combine judge score and audience score
        combined_score = judge_points[series, round, :] + audience_score_ranked

        competitors_combined_scores = combined_score[competitors_in_round]
        ranked_combined_scores = np.argsort(competitors_combined_scores)
        ranked_competitors = competitors_in_round[ranked_combined_scores]

        dance_off_array[ranked_competitors[0], i] = 1
        dance_off_array[ranked_competitors[1], i] = 1

    #compute probabilities

    dance_off_count = np.sum(dance_off_array, axis=1)
    dance_off_prop[competitors_in_round] = ((1.0 * dance_off_count[competitors_in_round]) / (1.0 * num_prob_iters)) * (1.0 - (2.0 * epsilon)) + epsilon

    # compute log probability of actual dance-off / not dance-off for each competitor

    # log_prob = 0.0
    #
    # for competitor in competitors_in_round:
    #
    #     if dance_off[series, round, competitor] == 0:
    #         prob = 1.0 - dance_off_prop[competitor]
    #     else:
    #         prob = dance_off_prop[competitor]
    #     log_prob += np.log(prob)

    dance_off_delta = np.absolute(np.subtract(competitor_in_dance_off[series, round, competitors_in_round]), dance_off_prop[competitors_in_round])
    ones_array = np.ones(competitors_in_round.shape[0])
    log_prob = np.sum(np.log(np.subtract(ones_array, dance_off_delta)))

    print log_prob

    return log_prob

def convert_score_to_points(competitor_scores_raw):

    '''take an input array of raw scores and return an array of ranked scores (points)'''

    points_array = np.zeros(competitor_scores_raw.shape[0])

    competitor_positions_low_high = np.argsort(competitor_scores_raw)

    print 'competitor_positions_low_high.shape', competitor_positions_low_high.shape

    competitor_positions_high_low = np.fliplr([competitor_positions_low_high])[0]

    print 'competitor_positions_high_low.shape', competitor_positions_high_low.shape

    # score to top ranked individual = number of competitors in the round
    active_score = np.count_nonzero(competitor_scores_raw)

    for i in range(competitor_positions_high_low.shape[0]):

        print competitor_positions_high_low

        print i, competitor_positions_high_low[i]



        competitor = competitor_positions_high_low[i]

        print 'competitor BNN', competitor



        competitor_score_raw = competitor_scores_raw[competitor]

        print 'competitor_score_raw', competitor_score_raw

        if competitor_score_raw == 0:
            points_array[competitor] = 0
        else:
            if i == 0:
                points_array[competitor] = active_score
            else:
                previous_competitor = competitor_positions_high_low[i-1]
                if competitor_score_raw == competitor_scores_raw[previous_competitor]:



                    points_array[competitor] = active_score
                else:
                    active_score -= 1
                    points_array[competitor] = active_score

        #print i, competitor, competitor_score_raw, points_array[competitor]

    return points_array


def normalise_vector_normal(vector, std_dev):

    '''Takes an input vector and normalises it to zero mean and given std_dev'''

    #handles true zeros as not present competitors and adjusts accordingly

    normalised_vector = np.zeros(vector.shape[0])

    non_zero_idx = np.nonzero(vector)
    vector_non_zeros = vector[non_zero_idx]
    mean = np.mean(vector_non_zeros)
    sd = np.std(vector_non_zeros)
    vector_non_zeros_norm = (vector_non_zeros - mean) / (1.0 * sd / std_dev)
    normalised_vector[non_zero_idx] = vector_non_zeros_norm

    return normalised_vector



def create_competitor_lists(judge_scores):

    series_list = []

    for series in range(num_series):
        for round in range(max_rounds):
            for competitor in range(max_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    series_list.append(series)

    series_set = sorted(set(series_list))

    print series_set

    series_rounds_dict = {}
    full_dict = {}

    for series in series_set:
        rounds_list = []
        for round in range(max_rounds):
            for competitor in range(max_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    rounds_list.append(round)
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

    return full_dict


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
judge_points = np.zeros((num_series, max_rounds, max_competitors))
popularity_score = np.zeros((num_series, max_rounds, max_competitors))
audience_points = np.zeros((num_series, max_rounds, max_competitors))
combined_points = np.zeros((num_series, max_rounds, max_competitors))
competitor_in_dance_off = np.zeros((num_series, max_rounds, max_competitors), dtype=int)
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
max_judge_score = np.ones((num_series, max_rounds))

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

competitor_in_dance_off[14][2][15] = 1
competitor_in_dance_off[14][2][10] = 1

competitor_in_dance_off[14][3][9] = 1
competitor_in_dance_off[14][3][14] = 1

competitor_in_dance_off[14][4][10] = 1
competitor_in_dance_off[14][4][12] = 1

competitor_in_dance_off[14][4][10] = 1
competitor_in_dance_off[14][4][12] = 1

competitor_in_dance_off[14][5][8] = 1
competitor_in_dance_off[14][5][11] = 1

competitor_in_dance_off[14][6][8] = 1
competitor_in_dance_off[14][6][10] = 1

competitor_in_dance_off[14][7][1] = 1
competitor_in_dance_off[14][7][9] = 1

competitor_in_dance_off[14][8][7] = 1
competitor_in_dance_off[14][8][8] = 1

competitor_in_dance_off[14][9][4] = 1
competitor_in_dance_off[14][9][7] = 1

competitor_in_dance_off[14][10][5] = 1
competitor_in_dance_off[14][10][6] = 1

competitor_in_dance_off[14][11][1] = 1
competitor_in_dance_off[14][11][5] = 1

competitor_in_dance_off[14][12][2] = 1
competitor_in_dance_off[14][12][4] = 1

competitor_in_dance_off[14][13][2] = 1
competitor_in_dance_off[14][13][3] = 1


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

series_round_competitor_dict = create_competitor_lists(judge_score)
series_competitor_dict = {}
series_round_dict = {}

print '** Rounds in each season'

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

series_round_competitor_dance_off_dict = create_competitor_lists(competitor_in_dance_off)

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

#test

for series, round_dict in series_round_competitor_dict.items():
    for round, competitor_list in round_dict.items():
        for competitor in competitor_list:
            print series, round, competitor, judge_score[series, round, competitor], "%.2f" % judge_score_normalised[series, round, competitor], judge_points[series, round, competitor]



# ********************************************************************************************************************
#
# Initialise competitor popularity

print 'Initialise competitor popularity'

for series, competitor_list in series_competitor_dict.items():
    for competitor in competitor_list:
        competitor_series_popularity_iterations[series, competitor, 0] = np.random.rand()
        print series, competitor, competitor_series_popularity_iterations[series, competitor, 0]

    print competitor_series_popularity_iterations[series, :, 0]

# ********************************************************************************************************************
#
# Initialise variable parameters

print 'Initialise variable parameters'

for parameter in range(num_variable_parameters):

    variable_parameters_iterations[parameter] = np.random.rand()
    print parameter, variable_parameters_iterations[parameter]



#********************************************************************************************************************
#
# Main program

for iteration in range(1, num_iters): #initial settings are in 0

















for series in series_list:
    for iteration in range(1, num_iters): #initial settings are in 0

        #initialise competitor popularity and variable parameters by copying values from the
        # previous iteration into this iteration
        competitor_series_popularity_iterations[series, :, iteration] = competitor_series_popularity_iterations[series, :, iteration - 1]
        variable_parameters_iterations[:, iteration] = variable_parameters_iterations[:, iteration - 1]

        competitor_popularity_live_base = competitor_series_popularity_iterations[series, :, iteration]
        variable_parameters_live = variable_parameters_iterations[:, iteration]

        for competitor in range(max_competitors):
            if valid_series_competitor[series, competitor] > 0:

                # compute log-likelihood of current settings

                series_log_like_base = 0

                for round in range(max_rounds):
                    if valid_series_round[series, round] > 0:
                        #then this was a scored round in that series
                        round_log_like = round_log_likelihood(series, round, competitor_popularity_live_base, variable_parameters_live)
                        series_log_like_base += round_log_like

                # sample an alternative popularity value for competitor

                this_competitor_popularity_base = competitor_popularity_live_base[competitor]
                this_competitor_popularity_variant = np.random.normal(this_competitor_popularity_base, comp_prob_norm_std)

                competitor_popularity_live_variant = competitor_popularity_live_base.copy
                competitor_popularity_live_variant[competitor] = this_competitor_popularity_variant

                # normalise the vector?



                # compute series log-likelihood for this variant

                series_log_like_variant = 0

                for round in range(max_rounds):
                    if valid_series_round[series, round] > 0:
                        # then this was a scored round in that series
                        round_log_like = round_log_likelihood(series, round, competitor_popularity_live_variant,
                                                              variable_parameters_live)
                        series_log_like_variant += round_log_like

                # sample existing OR variant based on probabilities

                base_prob_non_norm = np.exp(series_log_like_base)
                variant_prob_non_norm = np.exp(series_log_like_variant)
                total_prob = base_prob_non_norm + variant_prob_non_norm
                base_prob_norm = base_prob_non_norm / total_prob

                print 'base_prob_norm', base_prob_norm

                if np.random.rand() > base_prob_norm:
                    select_variant = True
                    competitor_popularity_live = competitor_popularity_live_variant.copy

                else:
                    select_variant = False
                    competitor_popularity_live = competitor_popularity_live_base.copy


        for variable_parameter in range(num_variable_parameters):

            series_log_like_base = 0

            for round in range(max_rounds):
                if valid_series_round[series, round] > 0:
                    # then this was a scored round in that series
                    round_log_like = round_log_likelihood(series, round, competitor_popularity_live,
                                                          variable_parameters_live)
                    series_log_like_base += round_log_like






        # normalise competitor popularity

        #loop over variable parameters and pick most likely
            #select alternative value for parameter
            #compute probability of both values
            #sample from those two probabilities and pick new value
            #update parameter value

#do the same process for the dance_score_weight which is series independent


# ********************************************************************************************************************
#
# Display distributions of popularity and compute expected values

