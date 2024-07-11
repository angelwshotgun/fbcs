import pandas as pd
import numpy as np
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

# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)

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

# Đọc dữ liệu từ GitHub
data = read_csv_from_github()

# Tách kết quả trận đấu và thông tin người chơi
y = data['Result']
X = data.drop('Result', axis=1)

# One-hot encoding cho dữ liệu người chơi
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Tên các feature sau khi encode
feature_names = encoder.get_feature_names_out(X.columns)

# Huấn luyện mô hình
model = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
model.fit(X_encoded, y)

def predict_win_probability(team, all_players):
    team_dict = {player: team_num for player, team_num in team}
    team_array = np.zeros((1, len(all_players)))
    
    for i, player in enumerate(all_players):
        if player in team_dict:
            team_array[0, i] = team_dict[player]
    
    team_df = pd.DataFrame(team_array, columns=all_players)
    team_encoded = encoder.transform(team_df)
    return model.predict_proba(team_encoded)[0][1]  # Xác suất đội 1 thắng

def split_teams(players):
    all_players = X.columns
    best_split = None
    min_diff = float('inf')
    
    for team1 in combinations(players, 5):
        team2 = tuple(set(players) - set(team1))
        team1_with_num = [(p, 1) for p in team1]
        team2_with_num = [(p, 2) for p in team2]
        prob1 = predict_win_probability(team1_with_num + team2_with_num, all_players)
        prob2 = 1 - prob1  # Xác suất đội 2 thắng
        diff = abs(prob1 - prob2)
        
        if diff < min_diff:
            min_diff = diff
            best_split = (team1, team2)
    
    return best_split

@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/players', methods=['GET'])
def get_players():
    return jsonify(list(X.columns))

@app.route('/divide_teams', methods=['POST'])
def divide_teams():
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

@app.route('/update_scores', methods=['POST'])
def update_scores():
    winning_team = request.json['winning_team']
    team1 = request.json['team1']
    team2 = request.json['team2']
    
    if winning_team == 'red':
        for player in team1:
            data.loc[data['Player'] == player, 'Result'] += 1
        for player in team2:
            data.loc[data['Player'] == player, 'Result'] -= 1
    else:
        for player in team2:
            data.loc[data['Player'] == player, 'Result'] += 1
        for player in team1:
            data.loc[data['Player'] == player, 'Result'] -= 1
    
    # Save the updated dataset to GitHub
    save_csv_to_github(data)
    return jsonify({'message': 'Cập nhật điểm thành công!'})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
