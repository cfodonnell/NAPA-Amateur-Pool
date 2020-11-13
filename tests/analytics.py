import pandas as pd
import numpy as np
import seaborn as sb
import base64
from io import BytesIO
from flask import send_file
from flask import request
import player_information as pi
import matplotlib
#matplotlib.use('Agg') # required to solve multithreading issues with matplotlib within flask
import matplotlib.pyplot as plt
import matplotlib.style as style
sb.set_context("talk", font_scale = 1)
style.use('seaborn-whitegrid')

#######################################################################
# napa Database structure
#######################################################################

#	TABLE			SCHEMA

#	team_a
#	team_b
#	lineup
#	all_perms
#	perm_score
    
def get_team_info():
    ''' If valid team IDs have been entered, then import player data from the NAPA website. If a random team has been selected,
     then create a random team using the create_rand_team function. If there is an error with the team IDs, or inconsistent team lengths have been chosen, then set error = True.'''
    
    error = False
    # Retrieve ids from the input forms
    A = [request.form.get('player_' +str(i) +'a') for i in range(1,6)]
    B = [request.form.get('player_' +str(i) +'b') for i in range(1,6)]
    rand_a = request.form.get('random_a')
    rand_b = request.form.get('random_b') 
    
    try:
        if (rand_a == '1') & (rand_b == '1'): # Two random teams
            (team_A_df, team_B_df) = pi.create_two_rand_teams(5)
        elif rand_a == '1': # Team A is random, team B is real
            team_A_df = pi.create_rand_team(5)
            team_B = [int(x) for x in B if x]
            team_B_df = pi.team_data(team_B)
            if len(team_A_df) != len(team_B_df):
                error = True
        elif rand_b == '1': # Team B is random, team A is real
            team_B_df = pi.create_rand_team(5)
            team_A = [int(x) for x in A if x]
            team_A_df = pi.team_data(team_A)
            if len(team_A_df) != len(team_B_df):
                error = True
        else: # Both teams are real
            team_A = [int(x) for x in A if x]
            team_B = [int(x) for x in B if x]
            team_A_df = pi.team_data(team_A)
            team_B_df = pi.team_data(team_B)
            if len(team_A_df) != len(team_B_df):
                error = True
    except:
        error = True
        return [], [], error
        
    return team_A_df, team_B_df, error

def load_team_a(con):
    ''' Select all players from team_a table and rename the columns. '''

    query = ''' SELECT * FROM team_a'''
    team_a = pd.read_sql_query(query,con)
    return team_a.rename(columns={'name': 'a_name', 'id':'a_id', 'eight_skill': 'a_skill'})
    
def load_team_b(con):
    ''' Select all players from team_b table and rename the columns. '''

    query = ''' SELECT * FROM team_b'''
    team_b = pd.read_sql_query(query,con)
    return team_b.rename(columns={'name': 'b_name', 'id':'b_id', 'eight_skill': 'b_skill'})
    
def get_prev_player(con, player_id, team):
    ''' Select the name and the ID of the player who was chosen on the previous webpage. '''
    
    query = '''
    SELECT name, id
    FROM ''' + team + '''
    WHERE id = ''' + str(player_id)
    
    player_name = pd.read_sql_query(query,con).values[0][0]
    player_id = pd.read_sql_query(query,con).values[0][1]
    
    return player_name, player_id
    
def update_lineup(con, player_name, player_id, cur_pos, poss):
    ''' Update the lineup table with the active matchups. Clear all later matchup entries to avoid webpage caching when using the back button. '''
    
    cursor = con.cursor()
    cursor.execute('''UPDATE lineup SET name = ''' + "'" + player_name + "'" + ''', id = ''' + str(player_id) + ''' WHERE pos = ''' + str(cur_pos) + ''';''')
    
    # poss is a list of all lineup entries to be cleared
    for pos in poss:
        cursor.execute('''UPDATE lineup SET id = 0 WHERE pos = ''' + str(pos) + ''';''')
            
    con.commit()
    cursor.close()
    
def add_prev_player(con, form, team, prev_pos, poss):
    ''' Add the player chosen on the previous webpage to the current lineup.'''
    
    player_id = request.form.get(form)
    
    query = '''
    SELECT name, id
    FROM ''' + team + '''
    WHERE id = ''' + str(player_id)
    
    player_name = pd.read_sql_query(query,con).values[0][0]
    player_id = pd.read_sql_query(query,con).values[0][1]
    
    update_lineup(con, player_name, player_id, prev_pos, poss)
    
    return player_name, player_id
    
def clear_lineup(con):
    '''Set all lineup ids equal to zero. '''
    
    cursor = con.cursor()
    cursor.execute('''UPDATE lineup SET id = 0 ;''')
    con.commit()
    cursor.close()

def get_lineup(con):
    ''' Return the current lineup table as a dataframe.'''

    query = ''' SELECT name, id FROM lineup ORDER BY pos '''
    
    return pd.read_sql_query(query,con).values
    
def get_short_lineup(con):
    ''' Return the current lineup table as a dataframe, ignoring empty entries.'''

    query = ''' SELECT name, id 
    FROM lineup
    WHERE id <> 0 
    ORDER BY pos '''
    
    return pd.read_sql_query(query,con).values
    
def load_team(con, team):
    ''' Return the remaining players available to be picked (i.e. those that have not been added to the lineup table).'''

    query = ''' SELECT * FROM team_'''+ team + ''' WHERE id NOT IN (SELECT id FROM lineup)'''
    team_df = pd.read_sql_query(query,con)
    return team_df.rename(columns={'name': team + '_name', 'id': team + '_id', 'eight_skill': team + '_skill'})
    
def print_figure_init(pj, cj):
    ''' Produce a bar plot visualization of the predicted match points for all permutations. Produce a swarmplot showing predicted raw score coefficients for each permutation.'''

    img = BytesIO()
    fig, axs = plt.subplots(figsize=(15,5), ncols=2)

    sb.despine(left=True)
    sb.countplot(x='r1', data=pj, color='darkred', ax=axs[0])
    axs[0].set_xlabel('Predicted winning points margin')
    axs[0].set_ylabel('Permutations')
    axs[0].set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in axs[0].get_xticklabels()])
    axs[0].xaxis.set_tick_params(rotation=45)
    g2 = sb.swarmplot(x=cj['r1'], color = 'darkred', size=10, ax=axs[1])
    axs[1].legend(loc='best', fontsize='small')
    axs[1].set_xlabel('Average winning probability')
    
    plt.savefig(img, format='png', bbox_inches = "tight")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
    
def print_figure(pj, cj):
    ''' Produce a bar plot visualization of the predicted match points for the remaining permutations. Produce a swarmplot showing predicted raw score coefficients for each permutation, with remaining possible permutations highlighted.'''

    img = BytesIO()
    fig, axs = plt.subplots(figsize=(15,5), ncols=2)

    sb.despine(left=True)
    sb.countplot(x='r2', data=pj, color='darkred', ax=axs[0])
    axs[0].set_xlabel('Predicted winning points margin')
    axs[0].set_ylabel('Permutations')
    axs[0].set_xticklabels(['{:.0f}'.format(float(t.get_text())) for t in axs[0].get_xticklabels()])
    axs[0].xaxis.set_tick_params(rotation=45)
    g2 = sb.swarmplot(x=cj['r1'], y=[""]*len(cj), hue=cj['round'], palette = ['lightgray', 'darkred'], size=10, ax=axs[1])
    axs[1].legend(loc='best', fontsize='small')
    axs[1].set_xlabel('Average winning probability')
    
    plt.savefig(img, format='png', bbox_inches = "tight")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
    
def get_clause(lineup):
    ''' Construct the SQL clause which will filter the remaining permutations based on the currently selected matchups. '''

    llen = len(lineup)
    clause = ''''''

    if llen >= 2: # i.e. if the lineup table contains one or more complete rounds
    
        clause = clause + '''SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[0][1]) + ''' AND id_b = ''' + str(lineup[1][1])
    
        if llen >= 4: # i.e. if the lineup table contains one or more complete rounds
            rnd = int(np.floor(llen/2) + 1) # current round
            for i in range(2,rnd):
                pos1 = 2*(i-1)
                pos2 = 2*(i-1)+1
            
                clause = clause + ''' INTERSECT SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[pos1][1]) + ''' AND id_b = ''' + str(lineup[pos2][1])
    
    return clause
    
def get_pick_clause(lineup):
    ''' Construct the SQL clause which will filter the remaining permutations based on the currently selected matchups. '''

    llen = len(lineup)
    clause = ''''''
    
    rnd = int(np.floor(llen/2) + 1) # current round
    for i in range(1,rnd):
        pos1 = 2*(i-1)
        pos2 = 2*(i-1)+1
            
        clause = clause + ''' INTERSECT SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[pos1][1]) + ''' AND id_b = ''' + str(lineup[pos2][1])
    
    return clause
    
def calc_stds_coef(con, team_A_df, team_B_df):
    ''' For each remaining possible permutation, find the average winning probability of each permutation containing each possible player matchup. The best choice for your team to put up is the player who has the lowest standard deviation across their matchups, i.e. regardless of who the opposing team chooses, the average winning probability for the subsequent remaining permutations will be approximately the same. '''
        
    lineup = get_short_lineup(con)
    clause = get_pick_clause(lineup)
        
    query = '''
    SELECT player_a, id_a, STDDEV_POP(avg_prob) as stddev_prob
    FROM (
    SELECT a.player_a, a.id_a, a.player_b, AVG(s.probability) as avg_prob
    FROM (
    SELECT permutation FROM all_perms ''' + clause + ''') as f 
    JOIN all_perms as a ON f.permutation = a.permutation
    JOIN perm_score as s ON a.permutation = s.permutation
    GROUP BY a.player_b, a.player_a, a.id_a ) as grouped_scores
    GROUP BY player_a, id_a
    HAVING id_a NOT IN (SELECT id FROM lineup)
    ORDER BY stddev_prob'''
    
    stds = pd.read_sql_query(query,con)
            
    return stds
    
def calc_coefs(con, team_A_df, player_b, player_b_id):
    ''' Find the average winning probability for all permutations containing the remaining players available on your team versus the player the opposition has chosen. The best choice for your team to put up is the player who has the highest average winning probability across all permutations where they play against the opposition's chosen player. Return the dataframe ranked in order of highest to lowest average winning probability.'''
    
    lineup = get_short_lineup(con)
    clause = get_pick_clause(lineup)
    
    query = '''
    SELECT a.id_a, a.player_a, a.player_b, AVG(s.probability) as avg_prob
    FROM (
    SELECT permutation FROM all_perms ''' + clause + ''') as f 
    JOIN all_perms as a ON f.permutation = a.permutation
    JOIN perm_score as s ON a.permutation = s.permutation
    WHERE a.id_b = ''' + str(player_b_id) + '''
    GROUP BY a.id_a, a.player_a, a.player_b
    ORDER BY avg_prob DESC '''
    
    team_A_df = pd.read_sql_query(query,con)
    
    return team_A_df
    
def agg_coefs_init(con):
    ''' Aggregate the winning probabilities from each permutation, returning their average values in a dataframe.''' 

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)

    query = '''
    SELECT permutation, probability
    FROM perm_score
    '''
    
    coef = pd.read_sql_query(query,con).values
    return pd.DataFrame(coef, columns = ['permutation','r1'])
    
def agg_scores_init(con):
    ''' Aggregate the match scores from each permutation, returning their total values in a dataframe.''' 

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)

    query = '''
    SELECT permutation, SUM(result)
    FROM all_perms
    GROUP BY permutation
    '''
    
    scores = pd.read_sql_query(query,con).values
    return pd.DataFrame(scores, columns = ['permutation','r1'])
    
def agg_coefs(con):
    ''' Aggregate the winning probabilities from each permutation, returning their average values in a dataframe. Furthermore, perform the aggregation on the remaining active permutations and add this an extra column.'''

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)
    
    query = '''
    SELECT r1.permutation, r1.avg_prob as r1_coef, r2.tot_score as r2_coef, CASE WHEN r2.tot_score IS NULL THEN 'All Predictions' ELSE 'Active Predictions' END as round
    FROM (SELECT permutation, probability as avg_prob FROM perm_score) as r1
    LEFT JOIN
    (SELECT s.permutation, s.probability as tot_score
    FROM (''' + clause + '''
    ) as p
    LEFT JOIN perm_score as s ON p.permutation = s.permutation) as r2 ON r1.permutation = r2.permutation
    '''
    
    coef_joined = pd.read_sql_query(query,con).values
    return pd.DataFrame(coef_joined, columns = ['permutation','r1','r2','round'])

def agg_scores(con):
    ''' Aggregate the match scores from each permutation, returning their total values in a dataframe. Futhermore, perform the aggregation on the remaining active permutations and add this an extra column.'''

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)
    
    query = '''
    SELECT r1.permutation, r1.sum as r1_score, r2.tot_score as r2_score
    FROM (SELECT permutation, SUM(result) as sum FROM all_perms
    GROUP BY permutation) as r1
    LEFT JOIN
    (SELECT a.permutation, SUM(a.result) as tot_score
    FROM (''' + clause + '''
    ) as p
    LEFT JOIN all_perms as a ON p.permutation = a.permutation
    GROUP BY a.permutation) as r2 ON r1.permutation = r2.permutation
    '''
    
    perms_joined = pd.read_sql_query(query,con).values
    return pd.DataFrame(perms_joined, columns = ['permutation','r1','r2'])
    
def count_perms(con):
    ''' Return the total number of permutations.'''
    
    query = '''
    SELECT COUNT(*) FROM perm_score
    '''
    return pd.read_sql_query(query,con).values[0][0]
    
def get_perm(con):
    ''' Return the active permutation id(s).'''
    
    lineup = get_short_lineup(con)
    clause = get_clause(lineup)
    query = clause
    
    return pd.read_sql_query(query,con).values[0]
    
def get_perm_coef(con, perm):
    ''' Return the active permutation winning probability.'''
    
    query = '''
    SELECT probability
    FROM perm_score
    WHERE permutation = ''' + str(perm)
    
    return pd.read_sql_query(query,con).values[0][0]
    
def get_average_coef(con):
    ''' Return the average winning probability over all permutations.'''
    
    query = '''
    SELECT AVG(probability)
    FROM perm_score
    '''
    return pd.read_sql_query(query,con).values[0][0]
    
def get_perm_rank(con, perm):
    ''' Return the rank of the permutation, ordered by decreasing winning probability.'''
    
    query = '''
    SELECT rank
    FROM
    (SELECT permutation, probability, RANK() OVER(ORDER BY probability DESC)
    FROM perm_score) as r
    WHERE permutation = ''' + str(perm)

    return pd.read_sql_query(query,con).values[0][0]
    
    
def final_lineup(con, perm):
    ''' Return the final lineup and the score coefficients for the individual matchups.'''
    
    query = '''
    SELECT player_a, probability, player_b FROM all_perms
    WHERE permutation = ''' + str(perm) + '''
    ORDER BY probability DESC
    '''
    
    return pd.read_sql_query(query,con).values
    
def a_pick_first(con):
    ''' When team a is picking first, use the calc_stds_coef function to determine the recommended player. Update which permutations are still active for the visualization.'''
        
    lineup = get_lineup(con)
    team_A_df = load_team(con, 'a')
    team_B_df = load_team(con, 'b')
        
    stds = calc_stds_coef(con, team_A_df, team_B_df)
    rec =['']*len(team_A_df)
    rec[0] = ' (Recommended)'
        
    pj = agg_scores(con)
    cj = agg_coefs(con)
    plot_url = print_figure(pj, cj)
    
    return lineup, stds, rec, plot_url

def a_pick_second(con, player_b, player_b_id):
    ''' When team a is picking second, use the calc_coefs function to determine the recommended player. Update which permutations are still active for the visualization.'''

    lineup = get_lineup(con)
    team_A_df = load_team(con, 'a')
        
    team_A_df = calc_coefs(con, team_A_df, player_b, player_b_id)
    rec =['']*len(team_A_df)
    rec[0] = ' (Recommended)'
        
    pj = agg_scores(con)
    cj = agg_coefs(con)
    plot_url = print_figure(pj, cj)
    
    return lineup, team_A_df, rec, plot_url


def b_pick(con):
    ''' When team b is picking, there is no need to return a recommended player. Return the current lineup and team b players available for selection, and update the visualization.'''
    
    team_B_df = load_team(con, 'b')
    lineup = get_lineup(con)
        
    pj = agg_scores(con)
    cj = agg_coefs(con)
        
    plot_url = print_figure(pj, cj)
    
    return lineup, team_B_df, plot_url

def create_summary(con):
    ''' Return summary statistics for the overall match and the final permutation, including the average total score coefficient over all permutations, the total score coefficient of the final permutation, and the rank of this permutation.'''

    pj = agg_scores(con)
    cj = agg_coefs(con)
    plot_url = print_figure(pj, cj)
        
    tot_perms = count_perms(con) #the total number of permutations (120 for 5 player teams)
    av_coef = "{0:.2f}".format(get_average_coef(con)) #the average total score coefficient over all permutations
    active_perm = get_perm(con)[0] #the id number of the final active permutation
    perm_coef = "{0:.2f}".format(get_perm_coef(con, active_perm)) #the total score coefficient of the final permutation
    rank = get_perm_rank(con, active_perm) #the rank of this permutation (lower rank number indicates higher total score coefficient)
        
    final = final_lineup(con, active_perm)
    
    return final, tot_perms, perm_coef, av_coef, rank, plot_url
    
def similar_skills(con, player_b_id):
    ''' Find the maximum score coefficient of all players on your team vs the chosen player on the opposing team.'''
    
    query = '''
    SELECT id, ABS(eight_skill - (SELECT eight_skill FROM team_b WHERE id = ''' + str(player_b_id) + '''))
    AS diffs
    FROM team_a
    WHERE id NOT IN (SELECT id FROM lineup)
    ORDER BY diffs
    LIMIT 1
    '''
    
    return pd.read_sql_query(query,con).values[0][0] 
