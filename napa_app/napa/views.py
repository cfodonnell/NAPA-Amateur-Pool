from flask import render_template
from flask import request
from napa import app
from napa import player_information as pi
from napa import analytics as an
import pandas as pd

####################################################
# Order of events (if team A picks first)
####################################################

#   EVENT				ROUTE

# 1. Load teams			/teams
# 2. Initialize database		/teams
# 3. Lineup table is cleared		/pick1
# 4. Team A first pick suggestion	/pick1
# 5. Player A1 added to lineup	/line1
# 6. Team B picks second		/line1
# 7. Player B1 added to lineup	/line2
# 8. Team B picks first		/line2
# 9. Player B2 added to lineup	/pick2
# 10. Team A second pick suggestion	/pick2
# 11. Player A2 added to lineup	/pick3
# 12. Team A third pick suggestion	/pick3
# 13. Player A3 added to lineup	/line3
# 14. Team B picks second		/line3
# 15. Player B3 added to lineup	/line4
# 15. Team B picks first		/line4
# 16. Player B4 added to lineup	/pick4
# 17. Team A fourth pick suggestion	/pick4
# 18. Player A4 added to lineup	/pick5
# 19. Team A fifth pick suggestion	/pick5
# 20. Player A5 added to lineup	/line5
# 21. Team B picks second		/line5
# 22. Produce match summary		/summary

####################################################
# Lineup table positions
####################################################

# As the match progresses, players are added to the lineup table. The indices (positions) at which they are entered are shown below:

# 0. Player A1
# 1. Player B1
# 2. Player A2
# 3. Player B2
# 4. Player A3
# 5. Player B3
# 6. Player A4
# 7. Player B4
# 8. Player A5
# 9. Player B5

    
@app.route('/')
def team_input():
    ''' Request which returns the input page.'''
    
    error_message = ''
    
    return render_template("input.html", error_message = error_message)
    
@app.route('/teams',methods=['GET', 'POST'])
def teams():
    ''' Request which imports the teams and initialized the database.'''
    
    # Import player information
    (team_A_df, team_B_df, error) = an.get_team_info()
    rand_a = request.form.get('random_a')
    if error == True:
        return render_template("input.html", error_message = 'Please enter valid NAPA player IDs, or choose the "create random team" option.')
    
    # Fit the model to generate predictions, and initialize database
    pi.init_db(team_A_df, team_B_df)
    
    # Combine team info into single table for easy viewing
    team_A_df = team_A_df.rename(columns={'Name': 'a_name', 'ID':'a_id', '8 Skill': 'a_skill'})
    team_B_df = team_B_df.rename(columns={'Name': 'b_name', 'ID':'b_id', '8 Skill': 'b_skill'})
    teams = pd.concat([team_A_df[['a_name','a_id','a_skill']],team_B_df[['b_name','b_id','b_skill']]],axis=1)
    
    return render_template("teams.html", teams = teams)

@app.route('/pick1',methods=['GET', 'POST'])
def pick1():
    ''' Player A1 (first pick).'''
    
    con = pi.open_sql_con()
    
    # Clear the lineup to prevent caching issues
    an.clear_lineup(con)

    # Load team data and identify the suggested player
    team_A_df = an.load_team_a(con)
    team_B_df = an.load_team_b(con)     
    stds = an.calc_stds_coef(con, team_A_df, team_B_df)
    rec =['']*len(team_A_df)
    rec[0] = ' (Recommended)'
    
    # Update analytics for visualization
    pj = an.agg_scores_init(con)
    cj = an.agg_coefs_init(con)  
    plot_url = an.print_figure_init(pj, cj)
        
    con.close()
        
    return render_template('pick1.html', teams1 = stds, rec = rec, plot_url = plot_url)

@app.route('/line1',methods=['GET', 'POST'])
def line1():
    ''' Player B1 (second pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by insterting the selected player on the previous page into the lineup table  
    id_1a = request.form.get('first_pick')
    (player_1a, player_1a_id) = an.get_prev_player(con, id_1a, 'team_a')
    an.update_lineup(con, player_1a, player_1a_id, 0, poss = [1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Load team and current lineup data
    team_B_df = an.load_team_b(con)    
    lineup = an.get_lineup(con)
    
    # Update analytics for visualization 
    pj = an.agg_scores_init(con)
    cj = an.agg_coefs_init(con)    
    plot_url = an.print_figure_init(pj, cj)
        
    con.close()
        
    return render_template('line1.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/line2',methods=['GET', 'POST'])
def line2():
    ''' Player B2 (first pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by inserting the selected player on the previous page into the lineup table
    an.add_prev_player(con, form = 'first_pick', team = 'team_b', prev_pos = 1, poss = [2, 3, 4, 5, 6, 7, 8, 9])
    # Retrieve the current lineup, available team B players, and the updated visualization
    (lineup, team_B_df, plot_url) = an.b_pick(con)
        
    con.close()
        
    return render_template('line2.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick2',methods=['GET', 'POST'])
def pick2():
    ''' Player A2 (second pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by insterting the selected player on the previous page into the lineup table 
    (player_2b, player_2b_id) = an.add_prev_player(con, form = 'second_pick', team = 'team_b', prev_pos = 3, poss = [2, 4, 5, 6, 7, 8, 9])
    # Retrieve the current lineup, available team A players, the recommended player selection, and the updated visualization
    (lineup, team_A_df, rec, plot_url) = an.a_pick_second(con, player_2b, player_2b_id)
        
    con.close()
        
    return render_template('pick2.html', options = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick3',methods=['GET', 'POST'])
def pick3():
    ''' Player A3 (first pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by insterting the selected player on the previous page into the lineup table
    an.add_prev_player(con, form = 'second_pick', team = 'team_a', prev_pos = 2, poss = [4, 5, 6, 7, 8, 9])
    # Retrieve the current lineup, available team A players, the recommended player selection, and the updated visualization
    (lineup, team_A_df, rec, plot_url) = an.a_pick_first(con)
        
    con.close()
        
    return render_template('pick3.html', teams3 = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/line3',methods=['GET', 'POST'])
def line3():
    ''' Player B3 (second pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by inserting the selected player on the previous page into the lineup table
    an.add_prev_player(con, form = 'third_pick', team = 'team_a', prev_pos = 4, poss = [5, 6, 7, 8, 9])
    # Retrieve the current lineup, available team B players, and the updated visualization
    (lineup, team_B_df, plot_url) = an.b_pick(con)
        
    con.close()
        
    return render_template('line3.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/line4',methods=['GET', 'POST'])
def line4():
    ''' Player B4 (first pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by inserting the selected player on the previous page into the lineup table
    an.add_prev_player(con, form = 'third_pick', team = 'team_b', prev_pos = 5, poss = [6, 7, 8, 9])
    # Retrieve the current lineup, available team B players, and the updated visualization
    (lineup, team_B_df, plot_url) = an.b_pick(con)
        
    con.close()
        
    return render_template('line4.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick4',methods=['GET', 'POST'])
def pick4():
    ''' Player A4 (second pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by insterting the selected player on the previous page into the lineup table 
    (player_4b, player_4b_id) = an.add_prev_player(con, form = 'fourth_pick', team = 'team_b', prev_pos = 7, poss = [6, 8, 9])
    # Retrieve the current lineup, available team A players, the recommended player selection, and the updated visualization
    (lineup, team_A_df, rec, plot_url) = an.a_pick_second(con, player_4b, player_4b_id)
        
    con.close()
        
    return render_template('pick4.html', options = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/pick5',methods=['GET', 'POST'])
def pick5():
    ''' Player A5 (first pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by insterting the selected player on the previous page into the lineup table 
    an.add_prev_player(con, form = 'fourth_pick', team = 'team_a', prev_pos = 6, poss = [8, 9])
    # Retrieve the current lineup, available team A players, the recommended player selection, and the updated visualization
    (lineup, team_A_df, rec, plot_url) = an.a_pick_first(con)
        
    con.close()
        
    return render_template('pick5.html', teams5 = team_A_df, rec = rec, lineup = lineup, plot_url = plot_url)
    
@app.route('/line5',methods=['GET', 'POST'])
def line5():
    ''' Player B5 (second pick).'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by inserting the selected player on the previous page into the lineup table
    an.add_prev_player(con, form = 'fifth_pick', team = 'team_a', prev_pos = 8, poss = [9])
    # Retrieve the current lineup, available team B players, and the updated visualization
    (lineup, team_B_df, plot_url) = an.b_pick(con)
        
    con.close()
        
    return render_template('line5.html', teamb = team_B_df, lineup = lineup, plot_url = plot_url )
    
@app.route('/summary',methods=['GET', 'POST'])
def summary():
    ''' Produce summary statistics for this permutation to see how well it compares to random selections.'''
    
    con = pi.open_sql_con()
    
    # Update the lineup by insterting the selected player on the previous page into the lineup table    
    an.add_prev_player(con, form = 'fifth_pick', team = 'team_b', prev_pos = 9, poss = [])
    # Retrieve the final lineup and summary statistics
    (final, tot_perms, perm_coef, av_coef, rank, plot_url) = an.create_summary(con)
        
    con.close()
        
    return render_template('summary.html', final = final, tot_perms = tot_perms, perm_coef = perm_coef, av_coef = av_coef, rank = rank,  plot_url = plot_url)
    
@app.route('/how',methods=['GET', 'POST'])
def how():
    ''' Welcome page and instructions for use.'''
        
    return render_template('how.html')
