import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
from flask import Flask, request, jsonify, send_from_directory
import os
import requests
from io import StringIO
from github import Github
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
model = None
encoder = None
feature_names = None
X = None
y = None

# GitHub configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = 'angelwshotgun/fbcs' 
FILE_PATH = 'match_data.csv'

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

def read_csv_from_github():
    file_content = repo.get_contents(FILE_PATH)
    file_data = file_content.decoded_content.decode('utf-8')
    return pd.read_csv(StringIO(file_data))

def save_csv_to_github(data):
    csv_content = data.to_csv(index=False)
    file = repo.get_contents(FILE_PATH)
    repo.update_file(FILE_PATH, "Update CSV file", csv_content, file.sha)

# Read data from GitHub
data = read_csv_from_github()

def train_model():
    global model, encoder, feature_names, X, y
    data = read_csv_from_github()
    y = data['Result']
    X = data.drop('Result', axis=1)
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    
    feature_names = encoder.get_feature_names_out(X.columns)
    
    base_model = LogisticRegression(solver='lbfgs')
    model = OneVsRestClassifier(base_model)
    model.fit(X_encoded, y)

def predict_win_probability(team, all_players):
    global model, encoder
    
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore')
    
    team_dict = {player: team_num for player, team_num in team}
    team_array = np.zeros((1, len(all_players)))
    
    for i, player in enumerate(all_players):
        if player in team_dict:
            team_array[0, i] = team_dict[player]
    
    team_df = pd.DataFrame(team_array, columns=all_players)
    team_encoded = encoder.transform(team_df)
    return model.predict_proba(team_encoded)[0][1]  # Probability of team 1 winning

def split_teams(players):
    global X
    
    if X is None:
        data = read_csv_from_github()
        y = data['Result']
        X = data.drop('Result', axis=1)
    
    all_players = X.columns
    best_split = None
    min_diff = float('inf')
    
    for team1 in combinations(players, 5):
        team2 = tuple(set(players) - set(team1))
        team1_with_num = [(p, 1) for p in team1]
        team2_with_num = [(p, 2) for p in team2]
        prob1 = predict_win_probability(team1_with_num + team2_with_num, all_players)
        prob2 = 1 - prob1  # Probability of team 2 winning
        diff = abs(prob1 - prob2)
        
        if diff < min_diff:
            min_diff = diff
            best_split = (team1, team2)
    
    return best_split

def get_player_scores():
    global model, encoder, feature_names, X, y

    if X is None:
        train_model()

    # Calculate individual player scores
    player_scores = {}
    for player in X.columns:
        # Create a team with the player and randomly selected teammates
        player_team = {p: 1 if p == player else 2 for p in random.sample(list(X.columns), 5)}

        # Encode the team
        team_df = pd.DataFrame([player_team], columns=X.columns)
        team_encoded = encoder.transform(team_df)

        # Predict win probability for the player's team
        win_prob = model.predict_proba(team_encoded)[0][1]

        # Calculate player score (win probability for the player's team)
        player_scores[player] = win_prob

    # Sort scores and return a list of tuples (player, score)
    sorted_scores = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/prediction')
def serve_manual_prediction():
    return send_from_directory(os.getcwd(), 'index2.html')

@app.route('/player')
def serve_player_scores_page():
    return send_from_directory(os.getcwd(), 'index3.html')

@app.route('/players', methods=['GET'])
def get_players():
    global X
    if X is None:
        train_model()
    return jsonify(list(X.columns))

@app.route('/divide_teams', methods=['POST'])
def divide_teams():
    try:
        players = request.json['players']
        team1, team2 = split_teams(players)
        prob1 = predict_win_probability([(p, 1) for p in team1] + [(p, 2) for p in team2], X.columns)
        prob2 = 1 - prob1
        return jsonify({
            'team1': list(team1),
            'team2': list(team2),
            'prob1': prob1,
            'prob2': prob2
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/player_scores', methods=['GET'])
def get_player_scores_api():
    player_scores = get_player_scores()
    return jsonify(player_scores)

@app.route('/retrain', methods=['GET'])
def retrain_model():
    train_model()
    return jsonify({'message': 'Model retrained successfully'})

@app.route('/reload_data', methods=['GET'])
def reload_data():
    global data, X, y, encoder, model, feature_names
    
    # Read data from GitHub
    data = read_csv_from_github()
    
    # Separate match results and player information
    y = data['Result']
    X = data.drop('Result', axis=1)
    
    # One-hot encoding for player data
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    
    # Get feature names after encoding
    feature_names = encoder.get_feature_names_out(X.columns)
    
    # Retrain the model
    model = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
    model.fit(X_encoded, y)
    
    return jsonify({'message': 'Data reloaded successfully!'})

@app.route('/update_scores', methods=['POST'])
def update_scores():
    winning_team = request.json['winning_team']
    team1 = request.json['team1']
    team2 = request.json['team2']
    
    # Get all player names from the dataframe columns
    all_players = data.columns.tolist()[:-1]  # Exclude 'Result' column
    
    # Create a dictionary to store values for each player
    player_values = {player: 0 for player in all_players}
    
    # Update values for players in team1
    for player in team1:
        player_values[player] = 1
    
    # Update values for players in team2
    for player in team2:
        player_values[player] = 2
    
    # Create a new row to add to the dataframe
    new_row = list(player_values.values())
    
    # Add the Result value
    if winning_team == 'red':
        new_row.append(1)  # Team 1 won
    else:
        new_row.append(2)  # Team 2 won
    
    # Add the new row to the dataframe
    data.loc[len(data)] = new_row
    
    # Save the updated dataframe to GitHub
    save_csv_to_github(data)
    
    # Reload data and retrain the model
    reload_data()
    
    return jsonify({'message': 'Scores updated and data reloaded successfully!'})


def initialize():
    global model, encoder, feature_names, X, y
    train_model()

# Set this function at the end of the file
if __name__ == '__main__':
    initialize()
    app.run(debug=True, port=int(os.environ.get('PORT', 8000)))
