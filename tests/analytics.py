import pandas as pd
import numpy as np
import seaborn as sb
import base64
from io import BytesIO
from flask import send_file
import matplotlib.pyplot as plt
import matplotlib.style as style
sb.set_context("talk", font_scale = 1)
style.use('seaborn-whitegrid')
    
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
    
def add_prev_player(con, form, team, prev_pos, poss):
    ''' Add the previously selected player to the current lineup.'''
    
    player_id = request.form.get(form)
    
    query = '''
    SELECT name, id
    FROM ''' + team + '''
    WHERE id = ''' + str(player_id)
    
    player_name = pd.read_sql_query(query,con).values[0][0]
    player_id = pd.read_sql_query(query,con).values[0][1]
    
    update_lineup(con, player_name, player_id, prev_pos, poss)
    
def update_lineup(con, player_name, player_id, cur_pos, poss):
    ''' Update the lineup table with the active matchups. Clear all later matchup entries to avoid webpage caching when using the back button. '''
    
    cursor = con.cursor()
    cursor.execute('''UPDATE lineup SET name = ''' + "'" + player_name + "'" + ''', id = ''' + str(player_id) + ''' WHERE pos = ''' + str(cur_pos) + ''';''')
    
    # poss is a list of all lineup entires to be cleared
    for pos in poss:
        cursor.execute('''UPDATE lineup SET id = 0 WHERE pos = ''' + str(pos) + ''';''')
            
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
    #sb.set_context("poster", font_scale = 1)
    #style.use('seaborn-whitegrid')
    sb.despine(left=True)
    sb.countplot(x='r1', data=pj, color='darkred', ax=axs[0])
    axs[0].set_xlabel('Predicted winning points margin')
    axs[0].set_ylabel('Permutations')
    g2 = sb.swarmplot(x=cj['r1'], color = 'darkred', size=10, ax=axs[1])
    axs[1].legend(loc='best', fontsize='small')
    axs[1].set_xlabel('Total score coefficient margin')
    
    plt.savefig(img, format='png', bbox_inches = "tight")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
    
def print_figure(pj, cj):
    ''' Produce a bar plot visualization of the predicted match points for the remaining permutations. Produce a swarmplot showing predicted raw score coefficients for each permutation, with remaining possible permutations highlighted.'''

    img = BytesIO()
    fig, axs = plt.subplots(figsize=(15,5), ncols=2)
    #sb.set_context("talk", font_scale = 1)
    #style.use('seaborn-whitegrid')
    sb.despine(left=True)
    sb.countplot(x='r2', data=pj, color='darkred', ax=axs[0])
    axs[0].set_xlabel('Predicted winning points margin')
    axs[0].set_ylabel('Permutations')
    g2 = sb.swarmplot(x=cj['r1'], y=[""]*len(cj), hue=cj['round'], palette = ['lightgray', 'darkred'], size=10, ax=axs[1])
    axs[1].legend(loc='best', fontsize='small')
    axs[1].set_xlabel('Total score coefficient margin')
    
    plt.savefig(img, format='png', bbox_inches = "tight")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
    
def get_clause(lineup):
    ''' Construct the SQL clause which will filter the remaining permutations based on the currently selected matchups. '''

    llen = len(lineup)
    clause = ''''''

    if llen >= 2:
    
        clause = clause + '''SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[0][1]) + ''' AND id_b = ''' + str(lineup[1][1])
    
        if llen > 3:
            rnd = int(np.floor(llen/2) + 1)
            for i in range(2,rnd):
                pos1 = 2*(i-1)
                pos2 = 2*(i-1)+1
            
                clause = clause + ''' INTERSECT SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[pos1][1]) + ''' AND id_b = ''' + str(lineup[pos2][1])
    
    return clause
    
def get_pick_clause(lineup):
    ''' Construct the SQL clause which will filter the remaining permutations based on the currently selected matchups. '''

    llen = len(lineup)
    clause = ''''''
    
    rnd = int(np.floor(llen/2) + 1)
    for i in range(1,rnd):
        pos1 = 2*(i-1)
        pos2 = 2*(i-1)+1
            
        clause = clause + ''' INTERSECT SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[pos1][1]) + ''' AND id_b = ''' + str(lineup[pos2][1])
    
    return clause
    
def calc_stds(con, team_A_df):
    ''' For each winning permutation, find the number of times a given player plays each member of the opposing team. The best choice for your team to put up is the player who has approximately equal frequencies of playing each player on the opposing team across all winning permutations, i.e. they have the lowest standard deviation in their frequency table. '''
    
    lineup = get_short_lineup(con)
    clause = get_pick_clause(lineup)
    stds = []
        
    for player in team_A_df['a_id'].values:
        query = '''
	SELECT STDDEV(player_freq)
	FROM (
	SELECT a.player_b, COUNT(a.id_b) as player_freq
	FROM (
	SELECT permutation FROM perm_score WHERE result > 0 ''' + clause + '''
	) as p
	LEFT JOIN all_perms as a ON p.permutation = a.permutation
	WHERE a.id_a = ''' + str(player) + ''' 
	GROUP BY a.player_b, a.id_b
	) as freqs
	'''
                
        stds.append(pd.read_sql_query(query,con).values[0][0])
            
    return stds
    
def calc_stds_coef(con, team_A_df, team_B_df):
    ''' For each winning permutation, find the number of times a given player plays each member of the opposing team. The best choice for your team to put up is the player who has approximately equal frequencies of playing each player on the opposing team across all winning permutations, i.e. they have the lowest standard deviation in their frequency table. '''
    
    lineup = get_short_lineup(con)
    clause = get_pick_clause(lineup)
    stds = []
        
    for player_a in team_A_df['a_id'].values:
    
        avg_tot_coefs = []
        
        for player_b in team_B_df['b_id'].values:
            query = ''' 
            SELECT AVG(s.score_coef) as avg_tot_coef
	    FROM (
	    SELECT permutation FROM all_perms 
	    WHERE id_a = ''' + str(player_a) + ''' AND id_b = ''' + str(player_b) + ''' ''' + clause + '''
	    ) as p
	    JOIN perm_score as s ON p.permutation = s.permutation
	    '''
                
            avg_tot_coefs.append(pd.read_sql_query(query,con).values[0][0])
        stds.append(np.std(np.array(avg_tot_coefs)))
        #stds.append(np.sum(np.array(avg_tot_coefs)))
            
    return stds
    
def calc_coefs(con, team_A_df, player_b, player_b_id):
    ''' Find the average total score coefficient for all permutations containing the remaining players available on your team versus the player the opposition has chosen. The best choice for your team to put up is the player who has the highest average total score coefficient across all permutations where they play against the opposition's chosen player. Return the dataframe ranked in order of highest to lowest average total score coefficient.'''
    
    lineup = get_short_lineup(con)
    clause = get_pick_clause(lineup)
    avgs = []
    
    for player in team_A_df['a_id'].values:
        
        query = '''
        SELECT AVG(tot_score)
        FROM (
        SELECT SUM(a.score_coef) as tot_score
        FROM (
        SELECT permutation FROM all_perms WHERE id_a = ''' + str(player) + ''' AND id_b = ''' + str(player_b_id) + ''' ''' + clause + '''
        ) as p
        LEFT JOIN all_perms as a ON p.permutation = a.permutation
        GROUP BY a.permutation ) as player_scores
        '''
            
        avgs.append(pd.read_sql_query(query,con).values[0][0])
        
    team_A_df['player_b'] = np.array([player_b]*len(team_A_df))
    team_A_df['avg_score_coef'] = np.array(avgs)
    team_A_df = team_A_df.sort_values(by='avg_score_coef', ascending = False)
    
    return team_A_df 
    
def calc_ind_coefs(con, team_A_df, player_b, player_b_id):
    ''' Find the maximum score coefficient of all players on your team vs the chosen player on the opposing team.'''
    
    lineup = get_short_lineup(con)
    clause = get_pick_clause(lineup)
    scores = []
    
    for player in team_A_df['a_id'].values:
        
        query = ''' SELECT score_coef FROM all_perms
        WHERE id_a = ''' + str(player) + ''' AND id_b = ''' + str(player_b_id)
            
        scores.append(pd.read_sql_query(query,con).values[0][0])
        
    team_A_df['player_b'] = np.array([player_b]*len(team_A_df))
    team_A_df['avg_score_coef'] = np.array(scores)
    team_A_df = team_A_df.sort_values(by='avg_score_coef', ascending = False)
    
    return team_A_df
    
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
       
    
def agg_coefs_init(con):
    ''' Aggregate the score coefficients from each permutation, returning their total values in a dataframe.''' 

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)

    query = '''
    SELECT permutation, SUM(score_coef)
    FROM all_perms
    GROUP BY permutation
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
    ''' Aggregate the score coefficients from each permutation, returning their total values in a dataframe. Futhermore, perform the aggregation on the remaining active permutations and add this an extra column.'''

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)

    query = '''
    SELECT r1.permutation, r1.sum as r1_coef, r2.tot_score as r2_coef, CASE WHEN r2.tot_score IS NULL THEN 'All Predictions' ELSE 'Active Predictions' END as round
    FROM (SELECT permutation, SUM(score_coef) FROM all_perms
    GROUP BY permutation) as r1
    LEFT JOIN
    (SELECT a.permutation, SUM(a.score_coef) as tot_score
    FROM (''' + clause + '''
    ) as p
    LEFT JOIN all_perms as a ON p.permutation = a.permutation
    GROUP BY a.permutation) as r2 ON r1.permutation = r2.permutation
    '''
    
    coef_joined = pd.read_sql_query(query,con).values
    return pd.DataFrame(coef_joined, columns = ['permutation','r1','r2','round'])

def agg_scores(con):
    ''' Aggregate the match scores from each permutation, returning their total values in a dataframe. Futhermore, perform the aggregation on the remaining active permutations and add this an extra column.'''

    lineup = get_short_lineup(con)
    clause = get_clause(lineup)
    
    query = '''
    SELECT r1.permutation, r1.sum as r1_score, r2.tot_score as r2_score
    FROM (SELECT permutation, SUM(result) FROM all_perms
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
    ''' Return the active permutation score coef.'''
    
    query = '''
    SELECT score_coef
    FROM perm_score
    WHERE permutation = ''' + str(perm)
    
    return pd.read_sql_query(query,con).values[0][0]
    
def get_average_coef(con):
    ''' Return the average score coef over all permutations.'''
    
    query = '''
    SELECT AVG(score_coef)
    FROM perm_score
    '''
    return pd.read_sql_query(query,con).values[0][0]
    
def get_perm_rank(con, perm):
    ''' Return the rank of the permutation, ordered by decreasing total score coef.'''
    
    query = '''
    SELECT rank
    FROM
    (SELECT permutation, score_coef, RANK() OVER(ORDER BY score_coef DESC)
    FROM perm_score) as r
    WHERE permutation = ''' + str(perm)

    return pd.read_sql_query(query,con).values[0][0]
    
    
def final_lineup(con, perm):
    ''' Return the final lineup and the score coefficients for the individual matchups.'''
    
    query = '''
    SELECT player_a, score_coef, player_b FROM all_perms
    WHERE permutation = ''' + str(perm) + '''
    ORDER BY score_coef DESC
    '''
    
    return pd.read_sql_query(query,con).values
    
def a_pick_first(con):

    id_2a = request.form.get('second_pick')
    (player_2a, player_2a_id) = an.get_prev_player(con, id_2a, 'team_a')
        
    poss = [4, 5, 6, 7, 8, 9]
    an.update_lineup(con, player_2a, player_2a_id, 2, poss)
        
    lineup = an.get_lineup(con)
    team_A_df = an.load_team(con, 'a')
    team_B_df = an.load_team(con, 'b')
        
    stds = an.calc_stds_coef(con, team_A_df, team_B_df)
    rec =['']*len(team_A_df)
    rec[stds.index(min(stds))] = ' (Recommended)'
        
    pj = an.agg_scores(con)
    cj = an.agg_coefs(con)
    plot_url = an.print_figure(pj, cj)
    
    return plot_url


#def b_pick_first:


#def a_pick_second:


#def b_pick_second:
