# Pool Team Captain Match Assistant
## Using machine learning, SQL, and Flask to gain a competitive advantage

Amateur pool league matches consist of two teams of players battling it out to win the most points as determined by that particular league's scoring system. Teams of 5 players in the North American Poolshooters Association (NAPA) play 5 consecutive individual matches, with each player playing exactly once. To promote inclusivity, the league uses an equalizing handicap system, where a lower ranked player requires less games to win a match than a higher ranked opponent. In each round, the team captain selects their player choice to compete against the selected player on the opposing team. The captains take turns to pick their player first, with a coin flip deciding who picks first in round 1.

The two main goals of this project were:
* Can pool match outcomes be predicted using machine learning, in spite of the equalizing handicap?
* Can the predictive outcomes be used to inform player selection strategies, in order to maximize a team's chances of victory?

The end product is a Flask web application which team captains can consult in-game to receive an informed suggestion of which player they should choose to play the next individual match.
Try the app at [www.magic8billiards.herokuapp.com] ! 

## Overview

Player information for both teams is entered by the user on the app homepage. Since there are 5 players on each team, each match has 120 possible permutations! These permutations are fed to a machine learning model which generates a table of predicted outcomes, stored in a central postgreSQL database. 
During the first round, the optimal first choice player is suggested to the user, who then has the option to select this player or choose another player from their team. After entering the player choice from the opposing team, this individual matchup is added to the database. Subsequently, there are are 24 (120/5) remaining possible permutations. For the second round, the optimal player is calculated based on these remaining permutations. The match continues in this fashion until all 5 players have been selected on each team.

![](images/app_schematic.png)

## File descriptions

* `napa_app` contains the files for running the Flask web application. See readme inside this folder for a detailed description of contents.
* `models` contains `model_selection.ipynb`, a notebook detailing the machine learning models that are called when using the application, along with the model pickle files. `old models` contains some inital exploration of NAPA data along with a PDF document describing some inital insights.
* `tests` contains `random_team_test.ipynb`, which is used to perform simulations of app usage with random team entries. The results of several tests using different player selection methods are also stored here.
