from itertools import permutations
import player_information as pi
import numpy as np
import pandas as pd
import pickle

class match:

    def __init__(self, team_A_df, team_B_df):
        
        team_A_vals = team_A_df[['Win %', '8 Skill', '8 Games', 'AvgPPM', 'State']].values
        team_B_vals = team_B_df[['Win %', '8 Skill', '8 Games', 'AvgPPM', 'State']].values

        perm = list(permutations(range(0,len(team_B_vals))))
        pms = [list(x) for x in perm]

        all_perms = []
        all_probs = []

        for pm_num, pm in enumerate(pms):

            team_B_mat = team_B_vals[pm]

            xa = np.zeros((team_A_vals.shape[0], team_A_vals.shape[1]+1))
            xb = np.zeros((team_A_vals.shape[0], team_A_vals.shape[1]+1))

            for j,dm in enumerate(team_A_vals):

                # Calculate the required races for each matchup based on the skill levels of the competing players
                a = team_A_vals[j]
                b = team_B_mat[j]
                a = np.insert(a, 0, pi.get_race(a[1],b[1])[0])
                b = np.insert(b, 0, pi.get_race(a[2],b[1])[1])
                xa[j,:] = a
                xb[j,:] = b

            matchup = np.zeros((team_A_vals.shape[0], team_A_vals.shape[1]+1))
            matchup[:,:-1] = xa[:,:-1] - xb[:,:-1]
            matchup[:,5] = xa[:,5]

            cols = ['Race Margin', 'Win % Margin', 'Skill Margin', 'Game Margin', 'AvgPPM Margin', 'State']
            lr_cols = ['Race Margin', 'Win % Margin', 'Skill Margin', 'State']
            matchup = pd.DataFrame(matchup, columns = cols)

            # filter the match dataframe to only contain the features required for logistic regression
            matchup = matchup[lr_cols]

            model = pickle.load(open('log.pkl', 'rb'))
            pred_prob = model.predict_proba(matchup)[:,0]
            prob_3win = pi.at_least_three(pred_prob)

            names_a = team_A_df['Name'].values
            names_b = team_B_df['Name'].values[pm]
            ids_a = team_A_df['ID'].values
            ids_b = team_B_df['ID'].values[pm]

            vec_int = np.vectorize(int)

            perm_df = pd.DataFrame({'permutation': np.array([pm_num]*len(team_A_df)), 'player_a':names_a, 'id_a': ids_a, 'race_a': vec_int(xa[:,0]),
                     'player_b': names_b, 'id_b': ids_b, 'probability': pred_prob})

            all_perms.append(perm_df)
            all_probs.append(prob_3win)

        all_perms_df = pd.concat(all_perms)
        uniques = all_perms_df.drop_duplicates(subset=['player_a','player_b'])
        
        # Create the match matrix for all possible player pairings

        match = np.zeros((5,5))
        self.a_names = uniques['player_a'].unique()
        self.b_names = uniques['player_b'].unique()
        
        for i, p_a in enumerate(self.a_names):
            for j, p_b in enumerate(self.b_names):
                match[i,j] = uniques[(uniques['player_a'] == p_a) & (uniques['player_b'] == p_b)]['probability'].values   
        
        self.match_matrix_= match
        self.means = np.zeros([5,5])
        self.all_probs_ = np.array(all_probs)
        self.curr_probs_ = np.array(all_probs)
        self.mean_prob_ = np.mean(all_probs)
        self.perms_ = pms
        self.a_ids = np.arange(5)
        self.b_ids = np.arange(5)
        self.a_choices = np.full(5,np.nan)
        self.b_choices = np.full(5,np.nan)
        self.current_round_ = 0
        self.best_perm_ = self.perms_[np.argmax(self.all_probs_)]
        self.best_perm_val_ = max(self.all_probs_)
        
        #initiliase a and b choices to nans
        #compare lengths of non-nan elements
        #if lengths are uneven, player is picking second
        #if lengths are even, player is picking first
        
    def filter_perms(self):
        
        perms_f = []
        dix = np.zeros(len(self.perms_))

        for i, pm in enumerate(self.perms_, start=1):
            if pm[int(self.a_choices[self.current_round_])] == int(self.b_choices[self.current_round_]):
                perms_f.append(pm)
                dix[i-1] = i

        scores_f = np.array([self.curr_probs_[int(i)-1] for i in dix if i > 0])

        team_a_new = np.setdiff1d(self.a_ids,self.a_choices[self.current_round_])
        team_b_new = np.setdiff1d(self.b_ids,self.b_choices[self.current_round_])

        self.perms_ = perms_f
        self.curr_probs_ = scores_f
        self.a_ids = team_a_new
        self.b_ids = team_b_new
        
    def set_round(self):
        
        num_choices_a = len(self.a_choices[~np.isnan(self.a_choices)])
        num_choices_b = len(self.b_choices[~np.isnan(self.b_choices)])
        
        if (num_choices_a > 0) & (num_choices_b > 0):
            if num_choices_a == num_choices_b:
                self.filter_perms()
                self.current_round_ += 1
                
    def picking_first(self):
        
        num_choices_a = len(self.a_choices[~np.isnan(self.a_choices)])
        num_choices_b = len(self.b_choices[~np.isnan(self.b_choices)])
        
        return num_choices_a == num_choices_b
        
    def a_pick(self, method):
        
        self.set_round()
        
        if self.picking_first():
        
            if method == 'minmax':

                pidx = np.zeros(len(self.perms_))
                means = np.zeros((len(self.a_ids),len(self.b_ids)))

                for idxb, b in enumerate(self.b_ids):      #looping though team b
                    for idxj, j in enumerate(self.a_ids):   #looping through team a
                        pidx0 = np.zeros(len(self.perms_))
                        for i, pm in enumerate(self.perms_, start=1):
                            if pm[j] == b:       #if player on team a is matched with player on team b
                                pidx0[i-1] = i      #find indices of perms containing this matchup

                        means[idxb,idxj] = np.mean(self.curr_probs_[pidx0>0])
                self.a_choices[self.current_round_] = self.a_ids[np.argmax(means.min(axis=0))]
                self.means = means

            elif method == 'random':
                self.a_choices[self.current_round_] = np.random.choice(self.a_ids)
                

            else:
                raise NameError('Please enter one of the following valid choices: best, random or greedy')
                
        else:
            if method == 'minmax':
    
                pidx0 = np.zeros(len(self.perms_))
                mean0 = []
                b_choice = self.b_choices[self.current_round_]

                for j in self.a_ids:   #looping through team a
                    pidx0 = np.zeros(len(self.perms_))
                    for i, pm in enumerate(self.perms_, start=1):
                        if pm[j] == b_choice:       #if player on team a is matched with b0
                            pidx0[i-1] = i      #find indices of perms containing this matchup

                    mean0.append(np.mean(self.curr_probs_[pidx0>0]))

                self.a_choices[self.current_round_] = self.a_ids[np.argmax(mean0)]
                self.means = mean0
                
            elif method == 'random':
                
                self.a_choices[self.current_round_] = np.random.choice(self.a_ids)
                
            elif method == 'greedy':
                
                b_choice = int(self.b_choices[self.current_round_])
                self.a_choices[self.current_round_] = np.argmax(self.match_matrix_[:,b_choice])

            else:
                raise NameError('Please enter one of the following valid choices: minmax, greedy, or random')
    
    def b_pick(self, method):
        
        self.set_round()
        
        if self.picking_first():
        
            if method == 'minmax':

                pidx = np.zeros(len(self.perms_))
                means = np.zeros((len(self.a_ids),len(self.b_ids)))

                for idxb, b in enumerate(self.b_ids):      #looping though team b
                    for idxj, j in enumerate(self.a_ids):   #looping through team a
                        pidx0 = np.zeros(len(self.perms_))
                        for i, pm in enumerate(self.perms_, start=1):
                            if pm[j] == b:       #if player on team a is matched with player on team b
                                pidx0[i-1] = i      #find indices of perms containing this matchup

                        means[idxb,idxj] = np.mean(self.curr_probs_[pidx0>0])
                self.b_choices[self.current_round_] = self.b_ids[np.argmin(means.max(axis=1))]

            elif method == 'random':
                self.b_choices[self.current_round_] = np.random.choice(self.b_ids)

            else:
                raise NameError('Please enter one of the following valid choices: best, random or greedy')
                
        else:
            if method == 'minmax':
    
                pidx0 = np.zeros(len(self.perms_))
                mean0 = []
                a_choice = self.a_choices[self.current_round_]

                for b in self.b_ids:   #looping through team a
                    pidx0 = np.zeros(len(self.perms_))
                    for i, pm in enumerate(self.perms_, start=1):
                        if pm[int(a_choice)] == b:       #if player on team a is matched with b0
                            pidx0[i-1] = i      #find indices of perms containing this matchup

                    mean0.append(np.mean(self.curr_probs_[pidx0>0]))

                self.b_choices[self.current_round_] = self.b_ids[np.argmin(mean0)]
                
            elif method == 'random':
                self.b_choices[self.current_round_] = np.random.choice(self.b_ids)

            else:
                raise NameError('Please enter one of the following valid choices: minmax, greedy, or random')
        
    def perm_ranks(self):
        
        ranked_probs = np.sort(self.all_probs_)[::-1]
        rank_df = pd.DataFrame({'rank':np.arange(1,len(self.all_probs_)+1),'prob':ranked_probs})
        return rank_df[rank_df['prob'].isin(self.curr_probs_)]['rank'].values
