import numpy as np
import player_information as pi

team_A = [53691, 60468, 61958, 53819, 61960]
team_B = [55440, 61943, 44891, 58517, 63803]
#team_A = [53691, 60468, 61958]
#team_B = [55440, 61943, 44891]

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

print(pi.get_lineup_one(team_A_df, team_B_df, 55440))

#print(pi.get_lineup(team_A_df, team_B_df, best_pm))
#print("Win A percentage: {}".format(num_a_wins/(num_a_wins + num_b_wins + num_tooclose)*100))
#print("Win B percentage: {}".format(num_b_wins/(num_a_wins + num_b_wins + num_tooclose)*100))
#print("Inconclusive result percentage: {}".format(num_tooclose/(num_a_wins + num_b_wins + num_tooclose)*100))
#print("Mean Score A: {}".format(mean_score_a))
#print("Mean Score B: {}".format(mean_score_b))
#print("Best Score A: {}".format(best_score))
#print("Mean Score Coef: {}".format(mean_score_coef))
#print("Best Score Coef: {}".format(best_score_coef))
#print("Percent Increase in Score Coef: {}".format(best_score_coef/mean_score_coef*100))


#Add case for player ID not found