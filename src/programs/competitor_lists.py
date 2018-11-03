__author__ = 'David'

import numpy as np

num_series = 20
max_num_competitors = 20
max_num_rounds = 20

judge_score_raw = np.zeros((num_series, max_num_rounds, max_num_competitors))

#populate with some data

judge_score_raw[14][2][1] = 54
judge_score_raw[14][3][1] = 35
judge_score_raw[14][4][1] = 39
judge_score_raw[14][5][1] = 36
judge_score_raw[14][6][1] = 32
judge_score_raw[14][7][1] = 34
judge_score_raw[14][8][1] = 35
judge_score_raw[14][9][1] = 38
judge_score_raw[14][10][1] = 40
judge_score_raw[14][11][1] = 36
judge_score_raw[14][12][1] = 77
judge_score_raw[14][13][1] = 119

judge_score_raw[14][2][2] = 63
judge_score_raw[14][3][2] = 36
judge_score_raw[14][4][2] = 36
judge_score_raw[14][5][2] = 35
judge_score_raw[14][6][2] = 30
judge_score_raw[14][7][2] = 38
judge_score_raw[14][8][2] = 38
judge_score_raw[14][9][2] = 40
judge_score_raw[14][10][2] = 45
judge_score_raw[14][11][2] = 38
judge_score_raw[14][12][2] = 76
judge_score_raw[14][13][2] = 116

judge_score_raw[14][2][3] = 63
judge_score_raw[14][3][3] = 31
judge_score_raw[14][4][3] = 33
judge_score_raw[14][5][3] = 33
judge_score_raw[14][6][3] = 35
judge_score_raw[14][7][3] = 39
judge_score_raw[14][8][3] = 37
judge_score_raw[14][9][3] = 38
judge_score_raw[14][10][3] = 45
judge_score_raw[14][11][3] = 37
judge_score_raw[14][12][3] = 73
judge_score_raw[14][13][3] = 116

judge_score_raw[14][2][4] = 56
judge_score_raw[14][3][4] = 36
judge_score_raw[14][4][4] = 30
judge_score_raw[14][5][4] = 32
judge_score_raw[14][6][4] = 36
judge_score_raw[14][7][4] = 33
judge_score_raw[14][8][4] = 36
judge_score_raw[14][9][4] = 36
judge_score_raw[14][10][4] = 39
judge_score_raw[14][11][4] = 37
judge_score_raw[14][12][4] = 73

judge_score_raw[14][2][5] = 52
judge_score_raw[14][3][5] = 27
judge_score_raw[14][4][5] = 27
judge_score_raw[14][5][5] = 29
judge_score_raw[14][6][5] = 32
judge_score_raw[14][7][5] = 33
judge_score_raw[14][8][5] = 33
judge_score_raw[14][9][5] = 33
judge_score_raw[14][10][5] = 31
judge_score_raw[14][11][5] = 31

judge_score_raw[14][2][6] = 44
judge_score_raw[14][3][6] = 24
judge_score_raw[14][4][6] = 16
judge_score_raw[14][5][6] = 18
judge_score_raw[14][6][6] = 26
judge_score_raw[14][7][6] = 27
judge_score_raw[14][8][6] = 25
judge_score_raw[14][9][6] = 23
judge_score_raw[14][10][6] = 24

judge_score_raw[14][2][7] = 53
judge_score_raw[14][3][7] = 32
judge_score_raw[14][4][7] = 28
judge_score_raw[14][5][7] = 24
judge_score_raw[14][6][7] = 26
judge_score_raw[14][7][7] = 32
judge_score_raw[14][8][7] = 31
judge_score_raw[14][9][7] = 32

judge_score_raw[14][2][8] = 62
judge_score_raw[14][3][8] = 31
judge_score_raw[14][4][8] = 31
judge_score_raw[14][5][8] = 32
judge_score_raw[14][6][8] = 33
judge_score_raw[14][7][8] = 34
judge_score_raw[14][8][8] = 31

judge_score_raw[14][2][9] = 57
judge_score_raw[14][3][9] = 30
judge_score_raw[14][4][9] = 33
judge_score_raw[14][5][9] = 0
judge_score_raw[14][6][9] = 36
judge_score_raw[14][7][9] = 32

judge_score_raw[14][2][10] = 50
judge_score_raw[14][3][10] = 27
judge_score_raw[14][4][10] = 27
judge_score_raw[14][5][10] = 30
judge_score_raw[14][6][10] = 25

judge_score_raw[14][2][10] = 50
judge_score_raw[14][3][10] = 27
judge_score_raw[14][4][10] = 27
judge_score_raw[14][5][10] = 30
judge_score_raw[14][6][10] = 25

judge_score_raw[14][2][11] = 49
judge_score_raw[14][3][11] = 27
judge_score_raw[14][4][11] = 31
judge_score_raw[14][5][11] = 24

judge_score_raw[14][2][12] = 46
judge_score_raw[14][3][12] = 25
judge_score_raw[14][4][12] = 24

judge_score_raw[14][2][13] = 57
judge_score_raw[14][3][13] = 31
judge_score_raw[14][4][13] = 0

judge_score_raw[14][2][14] = 55
judge_score_raw[14][3][14] = 28

judge_score_raw[14][2][15] = 45

# build lists



def create_competitor_lists(judge_scores):

    series_list = []

    for series in range(num_series):
        for round in range(max_num_rounds):
            for competitor in range(max_num_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    series_list.append(series)

    series_set = sorted(set(series_list))

    print series_set

    series_rounds_dict = {}
    full_dict = {}

    for series in series_set:
        rounds_list = []
        for round in range(max_num_rounds):
            for competitor in range(max_num_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    rounds_list.append(round)
        rounds_set = sorted(set(rounds_list))
        series_rounds_dict[series] = rounds_set

    for series, round_list in series_rounds_dict.items():
        round_competitor_dict = {}
        for round in round_list:
            print series, round
            competitor_list = []
            for competitor in range(max_num_competitors):
                if judge_scores[series][round][competitor] > 0:
                    #then this combination is valid
                    competitor_list.append(competitor)
            competitor_set = sorted(set(competitor_list))
            round_competitor_dict[round] = competitor_set
        full_dict[series] = round_competitor_dict

    return full_dict


series_round_competitor_dict = create_competitor_lists(judge_score_raw)

for series, round_dict in series_round_competitor_dict.items():
    for round, competitor_list in round_dict.items():
        print series, round, competitor_list

series = 14
round = 5
print series, round, series_round_competitor_dict[14][5]