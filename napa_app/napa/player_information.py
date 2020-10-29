from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import pickle
import copy
import psycopg2
import names
import subprocess
from itertools import permutations
from sqlalchemy import create_engine

class player_data:
    ''' Methods for scraping player data from NAPA website.'''
    
    off = 21 # offset for html tags
    
    def __init__(self, tags):
        self.tags = tags
        
    def general_info(self):
        '''Function to extract player details from their individual homepage.
        Returns a list of player details to be added to overall dataframe.'''
    
        player_name = str(self.tags[1].h2.a.string)[1:]
        player_id = int(self.tags[3].string)
        player_gen = str(self.tags[5].string)
        player_league = self.tags[7].string
        join_date = self.tags[9].string
        county = str(self.tags[11].string).split(',')[0]
        state = str(self.tags[11].string).split(',')[1]
        lg = self.tags[13].string
        if lg == 'Has not played':
            last_game = None
        else:
            last_game = lg
        allb = self.tags[17].find_all('a')
        
        if allb == 'INACTIVE':
            active_divisions = None
        else:
            active_divisions = [int(tg.string) for tg in allb]
            
        total_matches = float(self.tags[20].string)
        wins = float(str(self.tags[22].string).split('wins')[0])
        losses = total_matches-wins
        avgppm = float(self.tags[26].string)
        
        if total_matches == 0:
            win_percent = 0
        else:
            win_percent = wins/total_matches * 100
        
        return [player_name, player_id, player_gen, player_league, join_date, county, state, last_game, active_divisions,
            total_matches, wins, losses, avgppm, win_percent]
    
    def get_8_skills(self):
        '''This function extracts the skill level of each player for each 
        discipline from their Skill Levels tab, and returns the skills as a list.'''
        
        [tag.decompose() for tag in self.tags[self.off:52] if 'Class' in str(tag.text)]
        idx = [i for i, s in enumerate(self.tags[self.off:52]) if 'Ball' in str(s.text)]
        eightgames = str(self.tags[idx[1]+self.off].text).split(':')[1]
        eightskill = str(self.tags[idx[0]+self.off].text).split(':')[1]
    
        return [eightgames, eightskill]

def get_tags(add):
    '''Extract all relevant tags from the html.'''
    
    r = requests.get(add).text
    soup = BeautifulSoup(r, 'html.parser')
    return soup.find_all('td')      #all relevant data is contained within table tags

def encode_state(state):
    ''' Encode states to the numeric label required for the machine learning model.'''
    
    state = state.strip(' ')
    state_dict = {'Louisiana': 0, 'North Carolina': 1, 'Indiana': 2, 'Arkansas': 3, 'Colorado': 4,
                  'South Carolina': 5, 'Georgia': 6, 'Texas': 7, 'Kentucky': 8, 'Ohio': 9,
                  'Quebec': 10, 'Pennsylvania': 11, 'Kansas': 12, 'Florida': 13, 'Massachusetts': 14,
                  'California': 15, 'Washington': 16, 'Virginia': 17, 'Alabama': 18, 'Mississippi': 19,
                  'New Jersey': 20, 'Maryland': 21, 'Missouri': 22, 'New York': 23, 'Oklahoma': 24,
                  'Minnesota': 25}
    
    return state_dict[state]

def team_data(team):
    ''' The driveer function for performing the data extraction for each player on each team.'''
    
    team_lists = []
    cols = ['Name','ID','Gender','League','Join Date','County','State','Last Game','Active Divisions','Total Matches',
                  'Won','Lost','AvgPPM','Win %','8 Games','8 Skill']
    
    for pid in team:
        
        # retrieve data from player homepage
        add1 = 'https://www.napaleagues.com/stats.php?playerSelected=Y&playerID=' + str(pid)
        tags_gen = get_tags(add1)
        player_gen = player_data(tags_gen)
        # retrieve data from player skills tab
        add2 = 'https://www.napaleagues.com/stats.php?playerSelected=Y&playerID=' +str(pid) + '&xTab=20'
        tags_skill = get_tags(add2)
        player_skill = player_data(tags_skill)
        
        # check if player webpage is badly formatted or incomplete, if so return empty lists
        if 'wins' in str(tags_gen[4].string):
            return [None]*14
        elif tags_gen[3].find_all('strong') != []:
            return [None]*14
        elif tags_gen[3].find_all('font') != []:
            return [None]*14
        else:
            team_lists.append(player_gen.general_info() + player_skill.get_8_skills())
    
    # compile raw data into dataframe
    team_df = pd.DataFrame(team_lists, columns=cols)
    team_df['8 Skill'] = team_df['8 Skill'].apply(int)
    team_df['8 Games'] = team_df['8 Games'].apply(int)
    team_df['State'] = team_df['State'].apply(encode_state)

    return team_df

def load_model():
    return pickle.load(open('optimized_model_xgb.sav', 'rb'))

def get_race(pa, pb):
    ''' This function calculates the respective number of games that player 1 
    and player 2 are required to win, in order to win a particular match according 
    to NAPA rules, given their respective skill levels, p1 and p2.
    Full explanation at www.napaleagues.com/naparaces'''
    
    p1 = max([pa,pb])
    p2 = min([pa,pb])
    
    rc1 = [2,3,3,3,4,4,4,5,4,5,6,5,5,6,5,6,7,6,7,8,6,6,7,6,7,8,7,8,9,8,9,10]
    rc2 = [2,2,3,2,2,4,3,3,2,2,2,5,4,4,3,3,3,2,2,2,6,5,5,4,4,4,3,3,3,2,2,2]

    if (p1 < 40) & ((p1 - p2) < 20):
        r1 = rc1[0]
        r2 = rc2[0]
    elif (p1 < 40) & ((p1 - p2) > 19):
        r1 = rc1[1]
        r2 = rc2[1]
    elif (p1 > 39) & (p1 < 50) & ((p1 - p2) < 11):
        r1 = rc1[2]
        r2 = rc2[2]
    elif (p1 > 39) & (p1 < 50) & ((p1 - p2) > 10) & ((p1 - p2) < 27):
        r1 = rc1[3]
        r2 = rc2[3]
    elif (p1 > 39) & (p1 < 50) & ((p1 - p2) > 26):
        r1 = rc1[4]
        r2 = rc2[4]
    elif (p1 > 49) & (p1 < 70) & ((p1 - p2) < 7):
        r1 = rc1[5]
        r2 = rc2[5]
    elif (p1 > 49) & (p1 < 70) & ((p1 - p2) > 6) & ((p1 - p2) < 19):
        r1 = rc1[6]
        r2 = rc2[6]
    elif (p1 > 49) & (p1 < 70) & ((p1 - p2) > 18) & ((p1 - p2) < 30):
        r1 = rc1[7]
        r2 = rc2[7]
    elif (p1 > 49) & (p1 < 70) & ((p1 - p2) > 29) & ((p1 - p2) < 40):
        r1 = rc1[8]
        r2 = rc2[8]
    elif (p1 > 49) & (p1 < 70) & ((p1 - p2) > 39) & ((p1 - p2) < 49):
        r1 = rc1[9]
        r2 = rc2[9]
    elif (p1 > 49) & (p1 < 70) & ((p1 - p2) > 48):
        r1 = rc1[10]
        r2 = rc2[10]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) < 6):
        r1 = rc1[11]
        r2 = rc2[11]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 5) & ((p1 - p2) < 15):
        r1 = rc1[12]
        r2 = rc2[12]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 14) & ((p1 - p2) < 22):
        r1 = rc1[13]
        r2 = rc2[13]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 21) & ((p1 - p2) < 29):
        r1 = rc1[14]
        r2 = rc2[14]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 28) & ((p1 - p2) < 37):
        r1 = rc1[15]
        r2 = rc2[15]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 36) & ((p1 - p2) < 47):
        r1 = rc1[16]
        r2 = rc2[16]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 46) & ((p1 - p2) < 57):
        r1 = rc1[17]
        r2 = rc2[17]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 56) & ((p1 - p2) < 63):
        r1 = rc1[18]
        r2 = rc2[18]
    elif (p1 > 69) & (p1 < 90) & ((p1 - p2) > 62):
        r1 = rc1[19]
        r2 = rc2[19]
    elif (p1 > 89) & ((p1 - p2) < 5):
        r1 = rc1[20]
        r2 = rc2[20]
    elif (p1 > 89) & ((p1 - p2) > 4) & ((p1 - p2) < 12):
        r1 = rc1[21]
        r2 = rc2[21]
    elif (p1 > 89) & ((p1 - p2) > 11) & ((p1 - p2) < 18):
        r1 = rc1[22]
        r2 = rc2[22]
    elif (p1 > 89) & ((p1 - p2) > 17) & ((p1 - p2) < 23):
        r1 = rc1[23]
        r2 = rc2[23]
    elif (p1 > 89) & ((p1 - p2) > 22) & ((p1 - p2) < 29):
        r1 = rc1[24]
        r2 = rc2[24]
    elif (p1 > 89) & ((p1 - p2) > 28) & ((p1 - p2) < 36):
        r1 = rc1[25]
        r2 = rc2[25]
    elif (p1 > 89) & ((p1 - p2) > 35) & ((p1 - p2) < 43):
        r1 = rc1[26]
        r2 = rc2[26]
    elif (p1 > 89) & ((p1 - p2) > 42) & ((p1 - p2) < 49):
        r1 = rc1[27]
        r2 = rc2[27]
    elif (p1 > 89) & ((p1 - p2) > 48) & ((p1 - p2) < 59):
        r1 = rc1[28]
        r2 = rc2[28]
    elif (p1 > 89) & ((p1 - p2) > 58) & ((p1 - p2) < 69):
        r1 = rc1[29]
        r2 = rc2[29]
    elif (p1 > 89) & ((p1 - p2) > 68) & ((p1 - p2) < 75):
        r1 = rc1[30]
        r2 = rc2[30]
    elif (p1 > 89) & ((p1 - p2) > 74):
        r1 = rc1[31]
        r2 = rc2[31]
    else:
        r1 = 0
        r2 = 0
    
    if pa > pb:
        return r1, r2
    else:
        return r2, r1

def calc_score(preds, xa, xb):
    '''This function takes the predicted result margins, preds, and the team matrices,
    xa and xb, as inputs, and returns arrays containing the predicted number of games won
    for each team, along with the corresponding scores for each matchup calculated
    according to NAPA rules.'''
    
    tot_score = sum(preds)
    
    # find the cases where player a wins, and player b wins, respectively
    awins = copy.deepcopy(preds)
    # set indices where player a is predicted to lose to zero 
    awins[awins<0] = 0
    bwins = copy.deepcopy(preds)
    # set indices where player b is predicted to lose to zero
    bwins[bwins>0] = 0
    
    # convert the predicted score margin into individual predicted scores for player a and player b
    pred_score_a = xa[:,0] + bwins
    pred_score_b = xb[:,0] - awins
    
    a_points = np.zeros(len(awins))
    b_points = np.zeros(len(awins))
    
    # round the raw predicted game score coefficients to the nearest whole game and convert the predicted results into points using the NAPA scoring system
    a_points[(pred_score_a.round() == xa[:,0]) & (pred_score_b.round() == 0)] = 20
    a_points[(pred_score_a.round() == xa[:,0]) & (pred_score_b.round() != 0)] = 14
    a_points[(pred_score_a.round() == xa[:,0] - 1)] = 4
    a_points[(pred_score_a.round() < xa[:,0] - 1)] = 3
    a_points[(pred_score_a.round() == 0) & (pred_score_b.round() == xb[:,0])] = 1
    b_points[(pred_score_b.round() == xb[:,0]) & (pred_score_a.round() == 0)] = 20
    b_points[(pred_score_b.round() == xb[:,0]) & (pred_score_a.round() != 0)] = 14
    b_points[(pred_score_b.round() == xb[:,0] - 1)] = 4
    b_points[(pred_score_b.round() < xb[:,0] - 1)] = 3
    b_points[(pred_score_b.round() == 0) & (pred_score_a.round() == xb[:,0])] = 1
    
    for i in range(0, len(a_points)):
        if a_points[i] == b_points[i]:
            a_points[i] = 0
            b_points[i] = 0
    
    return tot_score, a_points, b_points, pred_score_a.round(), pred_score_b.round(), preds

def open_sql_con():
    '''Open connection to napa_db database.'''
    
    env = 'prod'
    
    if env = 'prod':

        database_url = subprocess.run(['heroku', 'config:get', 'DATABASE_URL', '--app', 'magic8billiards'],stdout=subprocess.PIPE).stdout
        
        return psycopg2.connect(database_url)
    else:
        with open('./sql_id.txt', 'r') as f:
            cred = [x.replace("'", '').strip() for x in f]
            dbname = cred[0]
            username = cred[1]
            pswd = cred[2]
    
        return psycopg2.connect(database = dbname, user = username, host='localhost', password=pswd)

def create_sql_engine():
    '''Create engine to write to napa_db database.'''
    
    env = 'prod'
    
    if env = 'prod':

        database_url = subprocess.run(['heroku', 'config:get', 'DATABASE_URL', '--app', 'magic8billiards'],stdout=subprocess.PIPE).stdout
        
        return create_engine(database_url)
    else:
        with open('./sql_id.txt', 'r') as f:
            cred = [x.replace("'", '').strip() for x in f]
            dbname = cred[0]
            username = cred[1]
            pswd = cred[2]
    
        return create_engine('postgresql://%s:%s@localhost/%s'%(username,pswd,dbname))
    
def create_two_rand_teams(num_players):
    ''' Create two random teams with parameters within typical ranges (to avoid id or name duplication).'''
    
    rwins = np.random.rand(num_players*2)*100
    rskills = np.random.randint(10,120,num_players*2)
    rgames = np.random.randint(0,100,num_players*2)
    rppm = rwins*0.14
    rstate = [np.random.randint(0,25)]*num_players*2
    rnames = [names.get_full_name() for i in range(0,num_players*2)]
    rids = np.random.choice(range(1,1000),num_players*2, replace=False)
    
    team1 = pd.DataFrame({'Name': rnames[0:num_players], 'ID': rids[0:num_players], 'Win %': rwins[0:num_players], '8 Skill': rskills[0:num_players],
     '8 Games': rgames[0:num_players], 'AvgPPM': rppm[0:num_players], 'State': rstate[0:num_players]})
    team2 = pd.DataFrame({'Name': rnames[num_players:num_players*2], 'ID': rids[num_players:num_players*2], 'Win %': rwins[num_players:num_players*2], '8 Skill': rskills[num_players:num_players*2],
     '8 Games': rgames[num_players:num_players*2], 'AvgPPM': rppm[num_players:num_players*2], 'State': rstate[num_players:num_players*2]})
    
    return team1, team2
                         
def create_rand_team(num_players):
    ''' Create a random team with parameters within typical ranges.'''
    
    rwins = np.random.rand(num_players)*100
    rskills = np.random.randint(10,120,num_players)
    rgames = np.random.randint(0,100,num_players)
    rppm = rwins*0.14
    rstate = [np.random.randint(0,25)]*num_players
    rnames = [names.get_full_name() for i in range(0,num_players)]
    rids = np.random.choice(range(1,1000),num_players, replace=False)
    
    return pd.DataFrame({'Name': rnames, 'ID': rids, 'Win %': rwins, '8 Skill': rskills, '8 Games': rgames,
                         'AvgPPM': rppm, 'State': rstate})

def init_db(team_A_df, team_B_df):
    ''' Generate the predicted results summary dataframe for the permutation with highest 
    likelihood of team A winning.'''
    
    team_A_vals = team_A_df[['Win %', '8 Skill', '8 Games', 'AvgPPM', 'State']].values
    team_B_vals = team_B_df[['Win %', '8 Skill', '8 Games', 'AvgPPM', 'State']].values
    
    perm = list(permutations(range(0,len(team_B_vals))))
    pms = [list(x) for x in perm]
    
    model = load_model()
    
    all_perms = []
    
    for pm_num, pm in enumerate(pms):
    
        team_B_mat = team_B_vals[pm]

        xa = np.zeros((team_A_vals.shape[0], team_A_vals.shape[1]+1))
        xb = np.zeros((team_A_vals.shape[0], team_A_vals.shape[1]+1))

        for j,dm in enumerate(team_A_vals):

            # Calculate the required races for each matchup based on the skill levels of the competing players
            a = team_A_vals[j]
            b = team_B_mat[j]
            a = np.insert(a, 0, get_race(a[1],b[1])[0])
            b = np.insert(b, 0, get_race(a[2],b[1])[1])
            xa[j,:] = a
            xb[j,:] = b

        matchup = np.zeros((team_A_vals.shape[0], team_A_vals.shape[1]+1))
        matchup[:,:-1] = xa[:,:-1] - xb[:,:-1]
        matchup[:,5] = xa[:,5]

        cols = ['Race Margin', 'Win % Margin', 'Skill Margin', 'Game Margin', 'AvgPPM Margin', 'State']
        matchup = pd.DataFrame(matchup, columns = cols)
        pred_res = model.predict(matchup)

        (score_coef, score_a, score_b, games_a, games_b, coefs) = calc_score(pred_res, xa, xb)

        names_a = team_A_df['Name'].values
        names_b = team_B_df['Name'].values[pm]
        ids_a = team_A_df['ID'].values
        ids_b = team_B_df['ID'].values[pm]

        vec_int = np.vectorize(int)
        
        perm_df = pd.DataFrame({'permutation': np.array([pm_num]*len(team_A_df)), 'player_a':names_a, 'id_a': ids_a, 'race_a': vec_int(xa[:,0]), 'predicted_games_a': games_a,
                 'predicted_points_a': score_a, 'predicted_points_b': score_b, 'predicted_games_b': games_b,
                 'race_b': vec_int(xb[:,0]), 'player_b': names_b, 'id_b': ids_b, 'score_coef': coefs})
        
        all_perms.append(perm_df)
    
    
    all_perms = pd.concat(all_perms)
    all_perms['result'] = all_perms['predicted_points_a'] - all_perms['predicted_points_b']
    perm_score = all_perms.groupby('permutation')['result'].sum() 
    perm_coefs = all_perms.groupby('permutation')['score_coef'].sum()
    perm_score = pd.merge(perm_score, perm_coefs, how='left',on='permutation')   
    engine = create_sql_engine()
    all_perms.to_sql('all_perms', engine, if_exists='replace')
    
    team_A_df = team_A_df[['Name', 'ID', '8 Skill']].rename(columns={'Name': 'name', 'ID':'id', '8 Skill': 'eight_skill'})
    team_B_df = team_B_df[['Name', 'ID', '8 Skill']].rename(columns={'Name': 'name', 'ID':'id', '8 Skill': 'eight_skill'})
    
    team_A_df.to_sql('team_a', engine, if_exists='replace')
    team_B_df.to_sql('team_b', engine, if_exists='replace')
    perm_score.to_sql('perm_score', engine, if_exists='replace')
    
    lineup = pd.DataFrame({'pos': np.arange(0,10), 'name': np.array(['']*10), 'id': np.zeros(10)})
    lineup.to_sql('lineup', engine, if_exists='replace')
    

