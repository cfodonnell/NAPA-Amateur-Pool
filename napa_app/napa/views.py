from flask import render_template
from flask import request
from napa import app
from napa import player_information as pi
from napa import analytics as an
import numpy as np
import pandas as pd
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sb

    
@app.route('/input')
def team_input():
    return render_template("input.html")
    
@app.route('/teams',methods=['GET', 'POST'])
def teams():
    
    if request.method == 'POST':
        A = [request.form.get('player_' +str(i) +'a') for i in range(1,6)]
        B = [request.form.get('player_' +str(i) +'b') for i in range(1,6)]
    	
    team_A = [int(x) for x in A if x]
    team_B = [int(x) for x in B if x]
    
    team_A_df = pi.team_data(team_A)
    team_B_df = pi.team_data(team_B)

    (best_pm, scores_A, scores_B, scores_margins, score_coefs) = pi.permute_match(team_A_df, team_B_df)

    best_score = max(scores_A)
    best_score_coef = max(score_coefs[scores_A == max(scores_A)])
    mean_score_coef = np.mean(score_coefs)
    mean_score_a = np.mean(scores_A)
    mean_score_b = np.mean(scores_B)
    num_a_wins = len(scores_margins[scores_margins>0])
    num_b_wins = len(scores_margins[scores_margins<0])
    num_tooclose = len(scores_margins[scores_margins==0])
    
    pi.get_all_perms(team_A_df, team_B_df)
    
    team_A_df = team_A_df.rename(columns={'Name': 'a_name', 'ID':'a_id', '8 Skill': 'a_skill'})
    team_B_df = team_B_df.rename(columns={'Name': 'b_name', 'ID':'b_id', '8 Skill': 'b_skill'})
    teams = pd.concat([team_A_df[['a_name','a_id','a_skill']],team_B_df[['b_name','b_id','b_skill']]],axis=1)
    players = teams['a_name'].values
    
    
    return render_template("teams.html", teams = teams, players = players)

@app.route('/pick1',methods=['GET', 'POST'])
def pick1():
    if request.method == 'POST':
    
        con = pi.open_sql_con()

        team_A_df = an.load_team_a(con)
        team_B_df = an.load_team_b(con)
        teams = pd.concat([team_A_df[['a_name','a_id','a_skill']],team_B_df[['b_name','b_id','b_skill']]],axis=1)
        
        stds = []
        
        for player in team_A_df['a_id'].values:
            query = '''
	    SELECT STDDEV(player_freq)
	    FROM (
	    SELECT a.player_b, COUNT(a.id_b) as player_freq
	    FROM (
	    SELECT permutation, result
	    FROM perm_score
	    WHERE result > 0 ) as p
	    LEFT JOIN all_perms as a ON p.permutation = a.permutation
	    WHERE a.id_a = ''' + str(player) + ''' 
	    GROUP BY a.player_b, a.id_b
	    ) as freqs
	    '''
                
            stds.append(pd.read_sql_query(query,con).values[0][0])
        
        rec =['']*len(team_A_df)
        rec[stds.index(min(stds))] = ' (Recommended)'
        
        pj = an.agg_scores_init(con)
        cj = an.agg_coefs_init(con)
        
        plot_url = an.print_figure_init(pj, cj)
        
        con.close()
        
    return render_template('pick1.html', teams1 = teams, rec = rec, plot_url = plot_url)

@app.route('/line1',methods=['GET', 'POST'])
def line1():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_1a = request.form.get('first_pick')
        (player_1a, player_1a_id) = an.get_prev_player(con, id_1a, 'team_a')

        poss = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        an.update_lineup(con, player_1a, player_1a_id, 0, poss)
        
        team_B_df = an.load_team_b(con)    
        lineup = an.get_lineup(con)
        
        pj = an.agg_scores_init(con)
        cj = an.agg_coefs_init(con)
        
        plot_url = an.print_figure_init(pj, cj)
        
        con.close()
        
    return render_template('line1.html', teamb = team_B_df, player_1a = lineup[0][0], plot_url = plot_url)
    
@app.route('/line2',methods=['GET', 'POST'])
def line2():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_1b = request.form.get('first_pick')
        (player_1b, player_1b_id) = an.get_prev_player(con, id_1b, 'team_b')
        
        poss = [2, 3, 4, 5, 6, 7, 8, 9]
        an.update_lineup(con, player_1b, player_1b_id, 1, poss)
 
        team_B_df = an.load_team(con, 'b')
        lineup = an.get_lineup(con)
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
    return render_template('line2.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick2',methods=['GET', 'POST'])
def pick2():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_2b = request.form.get('second_pick')
        (player_2b, player_2b_id) = an.get_prev_player(con, id_2b, 'team_b')
        
        poss = [2, 4, 5, 6, 7, 8, 9]
        an.update_lineup(con, player_2b, player_2b_id, 3, poss)
        
        lineup = an.get_lineup(con)
        team_A_df = an.load_team(con, 'a')
        
        avgs = []
        for player in team_A_df['a_id'].values:
        
            query = '''
            SELECT AVG(tot_score)
            FROM (
            SELECT SUM(a.score_coef) as tot_score
            FROM (
            SELECT permutation FROM all_perms WHERE id_a = ''' + str(player) + ''' AND id_b = ''' + str(player_2b_id) + ''' 
            INTERSECT
            SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[0][1]) + ''' AND id_b = ''' + str(lineup[1][1]) + ''' ) as p
            LEFT JOIN all_perms as a ON p.permutation = a.permutation
            GROUP BY a.permutation ) as player_scores
            '''
            
            avgs.append(pd.read_sql_query(query,con).values[0][0])
        
        team_A_df['player_b'] = np.array([player_2b]*len(team_A_df))
        team_A_df['avg_score_coef'] = np.array(avgs)
        team_A_df = team_A_df.sort_values(by='avg_score_coef', ascending = False)
        
        rec =['']*len(team_A_df)
        rec[0] = ' (Recommended)'
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
        
    return render_template('pick2.html', options = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick3',methods=['GET', 'POST'])
def pick3():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_2a = request.form.get('second_pick')
        (player_2a, player_2a_id) = an.get_prev_player(con, id_2a, 'team_a')
        
        poss = [4, 5, 6, 7, 8, 9]
        an.update_lineup(con, player_2a, player_2a_id, 2, poss)
        
        lineup = an.get_lineup(con)
        team_A_df = an.load_team(con, 'a')
        
        stds = []
        
        for player in team_A_df['a_id'].values:
            query = '''
	    SELECT STDDEV(player_freq)
	    FROM (
	    SELECT a.player_b, COUNT(a.id_b) as player_freq
	    FROM (
	    SELECT permutation FROM perm_score WHERE result > 0
	    INTERSECT
	    SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[2][1]) + ''' AND id_b = ''' + str(lineup[3][1]) + ''' 
	    INTERSECT
	    SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[0][1]) + ''' AND id_b = ''' + str(lineup[1][1]) + '''
	    ) as p
	    LEFT JOIN all_perms as a ON p.permutation = a.permutation
	    WHERE a.id_a = ''' + str(player) + ''' 
	    GROUP BY a.player_b, a.id_b
	    ) as freqs
	    '''
                
            stds.append(pd.read_sql_query(query,con).values[0][0])
        
        rec =['']*len(team_A_df)
        rec[stds.index(min(stds))] = ' (Recommended)'
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
    return render_template('pick3.html', teams3 = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/line3',methods=['GET', 'POST'])
def line3():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_3a = request.form.get('third_pick')
        (player_3a, player_3a_id) = an.get_prev_player(con, id_3a, 'team_a')
        
        poss = [5, 6, 7, 8, 9]
        an.update_lineup(con, player_3a, player_3a_id, 4, poss)
 
        lineup = an.get_lineup(con)
        team_B_df = an.load_team(con, 'b')
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
    return render_template('line3.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/line4',methods=['GET', 'POST'])
def line4():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_3b = request.form.get('third_pick')
        (player_3b, player_3b_id) = an.get_prev_player(con, id_3b, 'team_b')
        
        poss = [6, 7, 8, 9]
        an.update_lineup(con, player_3b, player_3b_id, 5, poss)
 
        lineup = an.get_lineup(con)
        team_B_df = an.load_team(con, 'b')
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
        
    return render_template('line4.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick4',methods=['GET', 'POST'])
def pick4():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_4b = request.form.get('fourth_pick')
        (player_4b, player_4b_id) = an.get_prev_player(con, id_4b, 'team_b')
        
        poss = [6, 8, 9]
        an.update_lineup(con, player_4b, player_4b_id, 7, poss)
        
        lineup = an.get_lineup(con)
        team_A_df = an.load_team(con, 'a')
        
        avgs = []
        for player in team_A_df['a_id'].values:
        
            query = '''
            SELECT AVG(tot_score)
            FROM (
            SELECT SUM(a.score_coef) as tot_score
            FROM (
            SELECT permutation FROM all_perms WHERE id_a = ''' + str(player) + ''' AND id_b = ''' + str(player_4b_id) + ''' 
            INTERSECT
            SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[0][1]) + ''' AND id_b = ''' + str(lineup[1][1]) + '''
            INTERSECT
            SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[2][1]) + ''' AND id_b = ''' + str(lineup[3][1]) + '''
            INTERSECT
            SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[4][1]) + ''' AND id_b = ''' + str(lineup[5][1]) + ''' ) as p
            LEFT JOIN all_perms as a ON p.permutation = a.permutation
            GROUP BY a.permutation ) as player_scores
            '''
            
            avgs.append(pd.read_sql_query(query,con).values[0][0])
        
        team_A_df['player_b'] = np.array([player_4b]*len(team_A_df))
        team_A_df['avg_score_coef'] = np.array(avgs)
        team_A_df = team_A_df.sort_values(by='avg_score_coef', ascending = False)
        
        rec =['']*len(team_A_df)
        rec[0] = ' (Recommended)'
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
        
    return render_template('pick4.html', options = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick5',methods=['GET', 'POST'])
def pick5():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_4a = request.form.get('fourth_pick')
        (player_4a, player_4a_id) = an.get_prev_player(con, id_4a, 'team_a')
        
        poss = [8, 9]
        an.update_lineup(con, player_4a, player_4a_id, 6, poss)
        
        lineup = an.get_lineup(con)
        team_A_df = an.load_team(con, 'a')
        
        stds = []
        
        for player in team_A_df['a_id'].values:
            query = '''
	    SELECT STDDEV(player_freq)
	    FROM (
	    SELECT a.player_b, COUNT(a.id_b) as player_freq
	    FROM (
	    SELECT permutation FROM perm_score WHERE result > 0
	    INTERSECT
	    SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[6][1]) + ''' AND id_b = ''' + str(lineup[7][1]) + ''' 
	    INTERSECT
	    SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[4][1]) + ''' AND id_b = ''' + str(lineup[5][1]) + ''' 
	    INTERSECT
	    SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[2][1]) + ''' AND id_b = ''' + str(lineup[3][1]) + ''' 
	    INTERSECT
	    SELECT permutation FROM all_perms WHERE id_a = ''' + str(lineup[0][1]) + ''' AND id_b = ''' + str(lineup[1][1]) + '''
	    ) as p
	    LEFT JOIN all_perms as a ON p.permutation = a.permutation
	    WHERE a.id_a = ''' + str(player) + ''' 
	    GROUP BY a.player_b, a.id_b
	    ) as freqs
	    '''
                
            stds.append(pd.read_sql_query(query,con).values[0][0])
        
        rec =['']*len(team_A_df)
        rec[stds.index(min(stds))] = ' (Recommended)'
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
    return render_template('pick5.html', teams5 = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/line5',methods=['GET', 'POST'])
def line5():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_5a = request.form.get('fifth_pick')
        (player_5a, player_5a_id) = an.get_prev_player(con, id_5a, 'team_a')
        
        poss = []
        an.update_lineup(con, player_5a, player_5a_id, 8, poss)
 
        lineup = an.get_lineup(con)
        team_B_df = an.load_team(con, 'b')
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        con.close()
        
    return render_template('line5.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url )
    
@app.route('/summary',methods=['GET', 'POST'])
def summary():
    if request.method == 'POST':
    
        con = pi.open_sql_con()
        
        id_5b = request.form.get('fifth_pick')
        (player_5b, player_5b_id) = an.get_prev_player(con, id_5b, 'team_b')
        
        poss = []
        an.update_lineup(con, player_5b, player_5b_id, 9, poss)
        
        pj = an.agg_scores(con)
        cj = an.agg_coefs(con)
        
        plot_url = an.print_figure(pj, cj)
        
        tot_perms = an.count_perms(con)
        av_coef = "{0:.2f}".format(an.get_average_coef(con))
        active_perm = an.get_perm(con)[0]
        perm_coef = "{0:.2f}".format(an.get_perm_coef(con, active_perm))
        rank = an.get_perm_rank(con, active_perm)
        
        final = an.final_lineup(con, active_perm)
        
        con.close()
        
    return render_template('summary.html', final = final, tot_perms = tot_perms, perm_coef = perm_coef, av_coef = av_coef, rank = rank,  plot_url = plot_url)
