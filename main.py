# code snippets for a meta tic tac toe bot

import numpy as np
import random
from math import sqrt
import torch
import torch.nn.functional as F
import torch.nn as nn

TRIALS = 257
EXPLORATION_FACTOR = 4 # chosen by fair dice roll. guaranteed to be random.

"""
The game is indexed as
0  1  2  9  10 11 18 19 20
3  4  5  12 13 14 21 22 23
6  7  8  15 16 17 24 25 26
27 28 29 36...
"""

TTT_LINES = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [0,3,6],
    [1,4,7],
    [2,5,8],
    [0,4,8],
    [2,4,6],
]

# game states (has the game finished?)
GAME_GOING = 0
FIRST_PLAYER_WIN = 1
SECOND_PLAYER_WIN = 2
DRAW = 3

# current turn states (whose turn is it?)
FIRST_PLAYER = 1
SECOND_PLAYER = 2

# move validity return (was the move you submitted a valid move?)
INVALID_MOVE = -1
# otherwise return a game state
# (GAME_GOING, FIRST_PLAYER_WIN, or SECOND_PLAYER_WIN)

POWERS_OF_3 = [3**i for i in range(100)]
HASH_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+="
if len(HASH_CHARS) < 64:
    print("WARNING: HASH_CHARS has incorrect length. This will result in an incorrect hashing function.")

class mt3_game():
    def __init__(self):
        self.board = np.zeros(81,dtype=int)   # the current board
        self.moveable = np.ones(9,dtype=int)  # which subsquares are playable in
        self.gameover = GAME_GOING  # whether the game is over
        self.moves = []
        self.current_player = FIRST_PLAYER
        self.board_metadata = np.zeros(9)

    def valid_moves(self):
        # returns valid cells to move in
        a = []
        for i in range(9):
            if self.moveable[i]:
                for j in range(9):
                    if self.board[9*i+j]:
                        a.append(9*i+j)
        return a

    def make_move(self, move):
        # checks if the move is valid
        if type(move) != type(0): return INVALID_MOVE
        if move<0 or move>80: return INVALID_MOVE
        if self.moveable[move//9] != 1: return INVALID_MOVE
        if self.board[move]!=0: return INVALID_MOVE
        if self.gameover != GAME_GOING:
            return INVALID_MOVE
        
        self.board[move] = self.current_player
        if self.current_player == FIRST_PLAYER:
            self.current_player = SECOND_PLAYER
        else:
            self.current_player = FIRST_PLAYER

        # checks if the game has concluded, and returns if so
        self.check_gameover(move)

        if self.gameover != GAME_GOING:
            self.moveable = np.zeros(9)
            return self.gameover

        # otherwise, configures gamestate for the next player, and returns 0
        b = move%9
        if self.board_metadata[b] != 0:
            self.moveable = np.array([i==0 for i in self.board_metadata],dtype=int)
        else:
            self.moveable = np.array([i==b for i in range(9)],dtype=int)
        return 0

    def check_gameover(self, move):
        c = (move//9)*9
        b = self.board[c:c+9]
        new_line = False
        for l in TTT_LINES:
            if 0 != b[l[0]] == b[l[1]] == b[l[2]]:
                self.board_metadata[move//9] = b[l[0]]
                new_line = True
                break
            
        if not new_line:
            for i in range(9):
                if b[i] == 0:
                    break
            else:
                self.board_metadata[move//9] = 3
                new_line = True
                
        b = self.board_metadata
        if new_line:
            for l in TTT_LINES:
                if 0 != b[l[0]] == b[l[1]] == b[l[2]]:
                    self.gameover = b[l[0]]
                    break
            else:
                for i in range(9):
                    if b[i] == 0:
                        break
                else: self.gameover = 3

    def flatten(self):
        # returns a "flattened" version of all the data
        p1 = [int(self.board[i] == FIRST_PLAYER) for i in range(81)]
        p2 = [int(self.board[i] == SECOND_PLAYER) for i in range(81)]
        m = [int(self.moveable[i//9]) for i in range(81)]
        if self.current_player == FIRST_PLAYER:
            return torch.tensor([p1, p2, m])
        elif self.current_player == SECOND_PLAYER:
            return torch.tensor([p2, p1, m])

    def pgn(self):
        # TODO make this return a standard-form pgn.
        return self.moves.copy

    def fen(self):
        s = 0
        for i in range(81):
            s += int(self.board[i])*POWERS_OF_3[i]
        for i in range(9):
            s += int(self.moveable[i])*POWERS_OF_3[i+81]
        if self.current_player == SECOND_PLAYER:
            s += POWERS_OF_3[90]
        r = ""
        for z in range(26): # magic constant chosen completely at random
            r += HASH_CHARS[s&63]
            s >>= 6
        return r

    def copy(self):
        G = mt3_game()
        G.board = self.board.copy()
        G.moveable = self.moveable.copy()
        G.gameover = self.gameover
        G.moves = self.moves.copy()
        G.current_player = self.current_player
        G.board_metadata = self.board_metadata.copy()
        return G

    def __str__(self):
        a = [[" "]*11 for _ in range(9)]
        a.insert(6,[""])
        a.insert(3,[""])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        a[4*l+j][4*k+i] = ".OX"[self.board[27*l+9*k+3*j+i]]

        return "\n".join(["".join(i) for i in a])
    
    def moveable_array(self):
        return torch.tensor([int(self.moveable[i//9]) for i in range(81)])
    
"""
class mt3_net(nn.Module):
    def __init__(self, dropout_rate=0):
        super(mt3_net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=12,
                               kernel_size=9, stride=9)
        # whatever, fix the architecture at some point
        self.linear2 = nn.Linear(12*9+81, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        self.linear5 = nn.Linear(512, 256)

        self.linear_p = nn.Linear(256, 81) #policy
        self.linear_v = nn.Linear(256, 1)  #value of position

        self.dropout_rate = dropout_rate
        self.f = nn.Flatten(start_dim=1)
    
    def forward(self,x):
        # x should have shape (batch_size * 3 * 81)
        a,b = torch.split(x,2,1)
        a = self.f(F.relu(self.conv1(a)))
        b = b.squeeze(1)
        x = torch.cat((a,b),1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))

        x = F.dropout(x, self.dropout_rate)

        p = F.relu(self.linear_p(x))
        v = torch.sigmoid(self.linear_v(x))

        p = F.softmax(p,1)

        return p,v
"""

#a smaller version to test on
class mt3_net_test(nn.Module):
    def __init__(self, dropout_rate=0):
        super(mt3_net_test, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32,
                               kernel_size=9, stride=9)
        self.linear2 = nn.Linear(32*9+81, 64)
        self.linear3 = nn.Linear(64, 64)

        self.linear_p = nn.Linear(64, 81) #policy
        self.linear_v = nn.Linear(64, 1)  #value of position

        self.dropout_rate = dropout_rate
        self.f = nn.Flatten(start_dim=1)
    
    def forward(self,x):
        # x should have shape (batch_size * 3 * 81)
        a,b = torch.split(x,2,1)
        a = self.f(F.relu(self.conv1(a)))
        b = b.squeeze(1)
        x = torch.cat((a,b),1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = F.dropout(x, self.dropout_rate)

        p = F.relu(self.linear_p(x))
        v = torch.sigmoid(self.linear_v(x))

        return p,v

def get_score(game, evaluation_net):
    """
    runs the net on the game and formats the data:
    returns two things:
    - a dictionary of possible move locations, with their policy
    (as predicted by the net)
    - the score, as predicted by the net. NOTE: if the game is over,
      then this prediction MUST be -1 or 1.
    """
    A = game.flatten().float().unsqueeze(0)
    policy, score = evaluation_net(A.to("cuda:0" if torch.cuda.is_available() else "cpu"))
    policy = nn.Softmax(dim=1)(policy)
    policy = policy[0]
    score = score[0]
    if game.gameover:
        if game.gameover == game.current_player:
            score = -1
        else:
            score = 1
    policy_dict = {}
    for i in range(9):
        if game.moveable[i]:
            for j in range(9):
                if game.board[9*i+j] == 0:
                    policy_dict[9*i+j] = float(policy[9*i+j])
    return policy_dict, float(score)

def weighted_choice(d, t):
    if t == 1:
        x = random.random()*sum(d.values())
        s = 0
        for k in d:
            s += d[k]
            if s>x: return k
        print("This code sould never run")
        return random.choice(list(d.keys()))
    else:
        x = random.random()*sum(i**t for i in d.values())
        s = 0
        for k in d:
            s += d[k]**t
            if s>=x: return k
        print("This code sould never run")
        return random.choice(list(d.keys()))        

"""
visited is a dictionary.
visited[n] is an array of the following values in order
- visits: how many times this has been visited
- a fixed tuple, of the predicted policy (P)
- a variable array, of the sum of scores if we make this move
- a variable array, of the number of times we visited each child node
"""

def update_single_position(visited, game, evaluation_net):
    global EXPLORATION_FACTOR, z
    if game.gameover != GAME_GOING:
        if game.gameover == 3: return 0
        return 1 if game.gameover == game.current_player else 0
    
    n = game.fen()
    
    if n not in visited:
        # initialise the unvisited node
        prior_policy, predicted_score = get_score(game, evaluation_net)
        visited[n] = [1,prior_policy,
                      {i:0 for i in prior_policy},
                      {i:0 for i in prior_policy}]
        return 1-predicted_score

    # else:
    N_sum,P,Q,N = visited[n]
    max_score = move = None
    for i in P:
        score=(0.5 if N[i]==0 else Q[i]/N[i])+EXPLORATION_FACTOR*P[i]*sqrt(N_sum)/(1+N[i])
        if max_score is None or score>max_score:
            max_score = score
            move = i
    
    if game.make_move(move)==-1:
        z = visited[n], game, move

    move_value = update_single_position(visited, game, evaluation_net)
    Q[move] += 2*move_value-1
    N[move] += 1

    return 1-move_value

def make_a_move(game, evaluation_net, temperature=1):
    global TRIALS, visited
    visited = {}
    for i in range(TRIALS):
        g = game.copy()
        update_single_position(visited, g, evaluation_net)
    policy = visited[game.fen()][3]
    move = weighted_choice(policy, temperature)
    return move, policy

def boN(model1, model2, game_class = mt3_game, N=5, t=4):
    a1=b1=c1=a2=b2=c2=0
    for _ in range(N):
        p1, p2 = model1, model2
        game = game_class()
        while game.gameover == GAME_GOING:
            d = {}
            for i in range(81):
                gt = game.copy()
                if gt.make_move(i) == -1: continue
                p,s = get_score(game, p1)
                d[i] = 1-s
            j = weighted_choice(d, t)
            game.make_move(j)
            p1,p2 = p2,p1
        if game.gameover == 1:
            a1 += 1
        if game.gameover == 2:
            b1 += 1
        if game.gameover == 3:
            c1 += 1
            
    for _ in range(N):
        p1, p2 = model2, model1
        game = game_class()
        while game.gameover == GAME_GOING:
            for i in range(81):
                gt = game.copy()
                if gt.make_move(i) == -1: continue
                p,s = get_score(game, p1)
                d[i] = 1-s
            j = weighted_choice(d, t)
            game.make_move(j)
            p1,p2 = p2,p1
        if game.gameover == 1:
            b2 += 1
        if game.gameover == 2:
            a2 += 1
        if game.gameover == 3:
            c2 += 1

    return a1,b1,c1,a2,b2,c2

def pretty_print_boN(model1, model2, N=5, t=4):
    a,b,c,d,e,f = boN(model1, model2, N, t)
    print(f"Model 1 going first: {a} wins, {b} losses, {c} draws.")
    print(f"Model 1 going second: {d} wins, {e} losses, {f} draws.")

def random_game(net, game_class = mt3_game, N=9):
    global game
    game = game_class()
    for i in range(N):
        game.make_move(weighted_choice(get_score(game,net)[0], 4))
    return game

def play_til_completion(game_class = mt3_game):
    game = game_class()
    while game.gameover == GAME_GOING:
        game.make_move(weighted_choice(get_score(game,H_)[0], 4))
    return game


def policy_dict_to_tensor(pd):
    return torch.tensor([(i in pd) and pd[i] for i in range(81)])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clear_data():
    global training_positions, labels_p, labels_v
    training_positions = torch.tensor([]).to(device)
    labels_p = torch.tensor([]).to(device)
    labels_v = torch.tensor([]).to(device)

def generate_data(evaluation_net, game_class, gen=1, batch_size=1):
    global training_positions, labels_p, labels_v
    p1_buffer = []
    p2_buffer = []

    for i in range(gen):
        #if gen%100==0:print(gen)
        game = game_class()
        while game.gameover == GAME_GOING:
            move, policy = make_a_move(game, evaluation_net)
            if game.current_player == 1:
                p1_buffer.append((game.flatten().float(), policy))
            else:
                p2_buffer.append((game.flatten().float(), policy))
            game.make_move(move)
        # now add the example data to TRAINING_INSTANCES
        game_result = [0,1,0,.5][int(game.gameover)]
        inputs = torch.stack([i[0] for i in p1_buffer]+
                             [i[0] for i in p2_buffer])
        label_policies = torch.stack(
            [policy_dict_to_tensor(i[1]) for i in p1_buffer]+
            [policy_dict_to_tensor(i[1]) for i in p2_buffer]).float()/(TRIALS-1)
        N,N2 = len(p1_buffer),len(p2_buffer)
        label_values = torch.tensor([(0.5*(N-i)+game_result*i)/N for i in range(1,N+1)]
                                     +[(0.5*(N2-i)+game_result*i)/N2 for i in range(1,N2+1)]).float()
        
        training_positions = torch.cat([training_positions, inputs.to(device)])
        labels_p = torch.cat([labels_p, label_policies.to(device)])
        labels_v = torch.cat([labels_v, label_values.to(device)])

def train(evaluation_net, epochs=1, batch_size=1):
    global training_positions, labels_p, labels_v, inputs, output_policies, label_policies, label_values, output_values
    if len(training_positions)==0:
        print("No positions!")
        return
    
    indices = np.arange(len(training_positions))
    
    criterion_policy = nn.CrossEntropyLoss()
    value_policy = nn.MSELoss()
    optimizer = torch.optim.SGD(evaluation_net.parameters(), lr=0.01, momentum=0.9)
    for it in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0
        for i in range(100):
            z = indices[i%(len(training_positions))]
            
            inputs = training_positions[z:z+1]
            label_policies = labels_p[z:z+1]
            label_values = labels_v[z:z+1]

            output_policies, output_values = evaluation_net(inputs.to(device))
            output_values = output_values.squeeze(1)

            loss1 = criterion_policy(output_policies, label_policies)
            loss2 = value_policy(output_values, label_values)

            loss = loss1 + 100*loss2
            
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            
            total_loss += loss.cpu().detach().numpy()
        if it%10 == 0: print(loss1, loss2)
    


H = mt3_net_test()
H_ = mt3_net_test()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
H.to(device)
H_.to(device)
# G = random_game(H_, ttt)
# G_ = mt3_game()

for i in range(100):
    print(f"Starting cycle {i}.")
    clear_data()
    generate_data(H, mt3_game, 20)
    train(H, 100)

# parameter count    
# sum(p.numel() for p in H.parameters() if p.requires_grad)

pretty_print_boN(H,H_,mt3_game,100)

"""
# to apply H to a game, use:

H(G.flatten().float().unsqueeze(0).to(device))

# to view a game, use:
G = random_game(H)
print(G)

# whether a game is over
G.gameover

pretty_print_boN(H,H_,ttt,100)

# attempts at other architectures. ultimately this ended up being worse

H = mt3_net_test2()
H.to(device)

class mt3_net_test2(nn.Module):
    def __init__(self, dropout_rate=0):
        super(mt3_net_test2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4,
                               kernel_size=9, stride=9)
        self.linear2 = nn.Linear(4*9+81, 16)

        self.linear_p = nn.Linear(16, 81) #policy
        self.linear_v = nn.Linear(16, 1)  #value of position

        self.dropout_rate = dropout_rate
        self.f = nn.Flatten(start_dim=1)
    
    def forward(self,x):
        # x should have shape (batch_size * 3 * 81)
        a,b = torch.split(x,2,1)
        a = self.f(F.relu(self.conv1(a)))
        b = b.squeeze(1)
        x = torch.cat((a,b),1)
        x = F.relu(self.linear2(x))

        x = F.dropout(x, self.dropout_rate)

        p = self.linear_p(x)
        v = torch.sigmoid(self.linear_v(x))

        p = F.softmax(p,1)

        return p,v

# attempt at batching games to expedite generation process.
# however, propagating Q-values up the monte carlo tree proved problematic to batch, and the attempt was aborted.

def get_scores(games, evaluation_net):
    ""
    runs the net on the game and formats the data:
    returns two things:
    - a dictionary of possible move locations, with their policy
    (as predicted by the net)
    - the score, as predicted by the net. NOTE: if the game is over,
      then this prediction MUST be -1 or 1.
    ""
    A = torch.stack(game.flatten() for game in games).float().to("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, score = evaluation_net(A)
    if game.gameover:
        if game.gameover == game.current_player:
            score = -1
        else:
            score = 1
    policy *= torch.stack(game.moveable_array() for game in games).float().to("cuda:0" if torch.cuda.is_available() else "cpu")
    return policy, float(score)

def weighted_choices(t):
    x = t.sum(dim=1)*torch.rand(len(t))
    s = 0
    C = t.cumsum(dim=1).unsqueeze(1)
    return torch.searchsorted(x,C).squeeze(1)

""
visited is a dictionary.
visited[n] is an array of the following values in order
- visits: how many times this has been visited
- a fixed tuple, of the predicted policy (P)
- a variable array, of the sum of scores if we make this move
- a variable array, of the number of times we visited each child node
""

def update_single_positions(visited, games, evaluation_net):
    global EXPLORATION_FACTOR
    if game.gameover != GAME_GOING:
        if game.gameover == 3: return 0
        return 1 if game.gameover == game.current_player else -1
    
    n = game.fen()
    
    if n not in visited:
        # initialise the unvisited node
        prior_policy, predicted_score = get_score(game, evaluation_net)
        visited[n] = [1,prior_policy,
                      {i:0 for i in prior_policy},
                      {i:0 for i in prior_policy}]
        return -predicted_score

    # else:
    N_sum,P,Q,N = visited[n]
    max_score = move = None
    for i in P:
        score=(N[i] and Q[i]/N[i])+EXPLORATION_FACTOR*P[i]*sqrt(N_sum)/(1+N[i])
        if max_score is None or score>max_score:
            max_score = score
            move = i
            
    
    if game.make_move(move)==-1:
        z = visited[n], game, move

    move_value = update_single_position(visited, game, evaluation_net)
    Q[move] += move_value
    N[move] += 1

    return -move_value

def make_a_move(game, evaluation_net, temperature=1):
    global TRIALS
    visited = {}
    for i in range(TRIALS):
        g = game.copy()
        update_single_position(visited, g, evaluation_net)
    policy = visited[game.fen()][3]
    move = weighted_choice(policy, temperature)
    return move, policy
"""
