import numpy as np
import pandas as pd

# Extracts rallies between a player and an opponent from a given CSV file
class Parser:
    def __init__(self, player, opponent, points_file, matches_file, cutoff):
        self.player = player
        self.opponent = opponent
        self.points_file = points_file
        self.matches_file = matches_file
        self.cutoff = cutoff

    def parse(self):
        # Read in CSV file
        points = pd.read_csv(self.points_file)

        # Obtain the player of interest's initials
        names = self.player.split("_")
        initials = names[0][0] + names[1][0]

        # Filter the dataframe by the player of interest and opponent of interest
        points = points[points["match_id"].str.contains(self.player)]
        points = points[points["match_id"].str.contains(self.opponent)]

        # Keep only the columns that will be used for rally parsing
        points = points[["match_id", "Pt", "Gm#", "TB?", "Serving", "1st", "2nd", "isSvrWinner", "Pts",
                         "Set1", "Set2", "Svr", "rallyCount"]]

        # Remove points that consist of two shots or less
        points["rallyCount"] = points["rallyCount"].astype(int)
        points = points[~(points.rallyCount <= 2)]

        # Remove points where the players last action is returning a serve
        points = points[~((points.rallyCount <= 3) & (points.Serving != initials))]
        points = points.drop(axis = 1, columns="rallyCount")
        
        # Only interested in non-tiebreaker points
        points = points[~(points["TB?"] == "S")]
        points["TB?"] = points["TB?"].astype(int)
        points = points[points["TB?"] == 0]
        points = points.drop(axis = 1, columns="TB?")

        # Sort points from earliest to latest (top to bottom)
        points = points.sort_values(by = ["match_id", "Pt"])
        
        matches = pd.read_csv(self.matches_file)
        matches = matches[["match_id", "Tournament", "Surface", "Best of"]]

        # Filter the points dataframe by the player of interest and opponent of interest
        matches = matches[matches["match_id"].str.contains(self.player)]
        matches = matches[matches["match_id"].str.contains(self.opponent)]

        # Only interested in points that occur after a certain date
        matches = matches[matches.match_id > self.cutoff]

        # Only interested in Grand slam and non-team ATP Tour matches
        matches.rename(columns = {"Tournament":"tournament"}, inplace = True)
        matches = matches[~matches.tournament.str.endswith(" CH")]
        # matches = matches[~matches.tournament.str.contains("Olympics")]
        # matches = matches[~matches.tournament.str.contains("Davis Cup")]
        matches = matches[~matches.tournament.str.endswith(" EXH")]
        matches = matches[~matches.tournament.str.contains("Itajai")]
        matches = matches[~matches.tournament.str.contains("Hopman Cup")]
        # matches = matches[~matches.tournament.str.contains("ATP Cup")]
        matches = matches[~matches.tournament.str.contains("NextGen Finals")]
        matches = matches[~matches.tournament.str.endswith(" Junior")]
        # matches = matches[~matches.tournament.str.contains("Laver Cup")]
        matches = matches[~matches.tournament.str.contains(" F[0-9]+", regex = True)]
        matches = matches.drop(axis = 1, columns="tournament")
        
        # Only interested in grass, clay and hard courts
        matches.rename(columns = {"Surface":"surface"}, inplace = True)
        matches = matches[~matches.surface.str.contains("Carpet")]
        matches["surface"] = matches["surface"].apply(clean_surfaces)

        points = pd.merge(points, matches, on = "match_id")

        # The number of the current game within a match, starting from 1
        game = None
        # The point number that refers to the start of the current game, starting from 1
        game_start_point = None
        # The rally string for the current point
        rally = None
        # Whether or not the player of interest is the server for the current point. True if they are, False otherwise
        is_server = False
        # The current point number within the current game, starting from 1
        point = None
        # The data labels, 0 for a lost point and 1 for a won point (from the perspective of the player of interest)
        y = []
        # The data to serve as input to the neural network, each element is a list of 1's and 0's
        x = []

        # The longest rally between the player and opponent
        max_rally = 0

        match_starts = [0]
        match_ids = [points.iloc[0]["match_id"][:8]]
        prev_match_id = points.iloc[0]["match_id"]

        surface_dict = {"clay": 0, "hard": 1, "grass": 2}

        score_dict = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}

        # Iterating through each row in the data
        for row in points.itertuples():
            # If a new game starts, the starting point and the server for that game needs to be set
            if int(row[3].split(" ")[0]) != game:
                game = int(row[3].split(" ")[0])
                game_start_point = int(row[2])
                is_server = (row[4] == initials)

            point = int(row[2]) - game_start_point + 1

            rally = row[6 - int(isinstance(row[6], float))]

            # Discarding some invalid rallies
            if (rally[-1] != "@") and (rally != "S") and (rally != "R") and not ((len(rally) == 2) and (rally[-1] == "#")):
                temp = get_input(remove_lets(rally), is_server, point)

                # If a rally is valid, include it in the ML input data
                if temp != [None]:
                    # Determining the starting and ending indices for each match
                    if row[1] != prev_match_id:
                        prev_match_id = row[1]
                        match_starts += [len(x)]
                        match_ids += [row[1][:8]]

                    # Court surface (80 - 82)
                    temp[80 + surface_dict[row[12]]] = 1

                    # Game Score (83 - 100)
                    scores = row[8].split("-")

                    if scores[0] == "AD":
                        if is_server:
                            temp[100] = 1
                        else:
                            temp[99] = 1

                    elif scores[1] == "AD":
                        if is_server:
                            temp[99] = 1
                        else:
                            temp[100] = 1

                    else:
                        if is_server:
                            temp[83 + (score_dict[scores[0]] * 4) + score_dict[scores[1]]] = 1
                        else:
                            temp[83 + (score_dict[scores[1]] * 4) + score_dict[scores[0]]] = 1

                    # Match score by sets (101 - 109)
                    if (is_server and row[11] == 1) or (not is_server and row[11] == 2):
                        temp[101 + (row[9] * 3) + row[10]] = 1
                    else:
                        temp[101 + (row[10] * 3) + row[9]] = 1

                    # Best of how ever many sets (110)
                    temp[110] = int(row[13] == 5)

                    x += [temp]
                    y += [int((row[7] == 1 and is_server) or (row[7] == 0 and not is_server))]
        
        # Normalising rally length (79)
        x = np.array(x, dtype = float)
        mu = np.mean(x[:, 79])
        sigma = np.std(x[:, 79])
        x[:, 79] = (x[:, 79] - mu) / sigma
        x = x.tolist()

        return x, y


# Extract the input data for the neural network from a given rally string
# rally - the rally string for the current point
# is_server - True if the player of interest is serving for the given rally, False otherwise
# point - the current point number within the current game, starting from 1
def get_input(rally, is_server, point):
    # The start and end indices for each shot within the rally string
    start_indices = []
    end_indices = []
    # The two most recent positions of the server (most recent position is the first list element)
    server_positions = None
    # The two most recent positions of the returner (most recent position is the first list element)
    returner_positions = None
    # The previously played shot
    previous_shot = None
    # The shot currently being played
    current_shot = None
    # The direction of the shot currently being played
    current_direction = None
    # The direction of the previously played shot
    prev_direction = None

    cur_shot_ex = None
    prev_shot_ex = None

    # The server always serves from the same position
    server_positions = [[1, 0], [1, 0]]

    direction_count = 0
    ended_early = 0

    for i in range(len(rally)):
        # The serve direction character
        if rally[i] in ["4", "5", "6"]:
            start_indices += [i]

        # Update the server's position if they approach the net off the serve
        elif rally[i] == "+" and i == 1:
            server_positions = [[1, 0], [1, 1]]

        # Any following shot character
        elif rally[i] in ["f", "b", "r", "s", "v", "z", "o", "p", "u", "y", "l", "m", "h", "i", "j", "k", "t", "q"]:
            start_indices += [i]
            end_indices += [i]

    # Discard the rally if it is just the opponent of interest serving
    if (len(start_indices) <= 1) and not (is_server):
        return [None]

    end_indices += [len(rally)]
    ball = None
    approach = False

    # Returner's position depends on the side being served from
    returner_positions = [[(point % 2) * 2, 0]] * 2

    # Ball trajectory post of a serve also depends on the side being served from
    if rally[start_indices[0]: end_indices[0]] == "6": 
        ball = [1, 1, 1]

    else:
        ball = [(point % 2) * 2] * 3

    # Enumerating all shots following the serve
    for i in range(1, len(start_indices)):
        approach = False
        depth = -1

        # Shots taking place at the net
        if rally[start_indices[i]] in ["v", "z", "o", "p", "h", "i", "j", "k"]:
            depth = 2
            previous_shot = current_shot
            current_shot = rally[start_indices[i]]

        # Shots taking place at the baseline
        elif rally[start_indices[i]] in ["f", "b", "r", "s", "u", "y", "l", "m", "t"]:
            depth = 0
            previous_shot = current_shot
            current_shot = rally[start_indices[i]]

        direction = False
        mod = False
        # Enumerating all information describing a single shot
        for j in range(start_indices[i] + 1, end_indices[i]):

            # Obtain the direction of a shot
            if rally[j] in ["1", "2", "3"]:
                direction = True
                direction_count += 1
                prev_direction = current_direction
                current_direction = int(rally[j])

            # Discard the rally if any shot doesn't have a direction
            if (j == end_indices[i] - 1) and (not direction):
                return [None]

            # Court position information
            if rally[j] in ["+", "-", "="]:
                # Approach shot
                if rally[j] == "+":
                    approach = True
                    mod = True
                    depth += 1
                    prev_shot_ex = cur_shot_ex
                    cur_shot_ex = 0

                    if depth > 2: 
                        depth -= 1

                # Shot taking place at the net
                elif rally[j] == "-": 
                    mod = True
                    depth = 2

                # Shot taking place at the baseline
                else: 
                    mod = True
                    depth = 0

            if rally[j] == "^":
                prev_shot_ex = cur_shot_ex
                cur_shot_ex = 1

            if rally[j] == ";":
                prev_shot_ex = cur_shot_ex
                cur_shot_ex = 2
                
        if not mod:
            prev_shot_ex = cur_shot_ex
            cur_shot_ex = None

        # Not sure about this honestly
        if (not is_server and len(start_indices) % 2 == 1) or (is_server and len(start_indices) % 2 == 0):
            if not direction:
                return [None]

        col = ball[depth]

        # Updating server's position
        if i % 2 == 0:
            update_position(server_positions, approach, col, depth)

        # Updating returner's position
        else:
            update_position(returner_positions, approach, col, depth)
        
        # Updating the ball's trajectory
        if current_direction == None:
            return [None]
        ball = [3 - current_direction] * 3

        if (depth == 0) or (col == 1):
            pass     

        elif depth == 1:
            ball[0] += int(current_direction == 3)
            ball[0] -= int(col / 2) * int(current_direction != 2)

        else:
            ball[0] += int(current_direction > 1)
            ball[0] -= int(col / 2) * (int(current_direction == 2) + 1)

        # Not sure about this honestly
        if i == len(start_indices) - 2: 
            if (not is_server and len(start_indices) % 2 == 1) or (is_server and len(start_indices) % 2 == 0):
                if not direction:
                    return [None]

                ended_early = 1
                break

    # Discard a rally if not every shot in the extracted sequence had a direction
    if direction_count != len(start_indices) - 1 - ended_early:
        return [None]

    if (current_shot == "t") or (current_shot == "q") or (previous_shot == "t") or (previous_shot == "q"):
        return [None]

    if previous_shot is None:
        return [None]

    # Mapping each shot to an index in the output
    shot_dict = {"f": 0, "b": 1, "r": 2, "s": 3, "v": 4, "z": 5, "o": 6, "p": 7, "u": 8, "y": 9, "l": 10, "m": 11,
                "h": 12, "i": 13, "j": 14, "k": 15}

    # Formatting and setting the output
    x = [0] * 111

    # When the player of interest is serving
    if is_server:
        # Current player position (0 - 8)
        x[(server_positions[0][0] * 3) + server_positions[0][1]] = 1
        # Previous player position (9 - 17)
        x[9 + (server_positions[1][0] * 3) + server_positions[1][1]] = 1

        # Current opponent position (39 - 47)
        x[39 + (returner_positions[0][0] * 3) + returner_positions[0][1]] = 1
        # Previous opponent position (48 - 56)
        x[48 + (returner_positions[1][0] * 3) + returner_positions[1][1]] = 1

    # When the opponent of interest is serving
    else:
        # Current player position (0 - 8)
        x[(returner_positions[0][0] * 3) + returner_positions[0][1]] = 1
        # Previous player position (9 - 17)
        x[9 + (returner_positions[1][0] * 3) + returner_positions[1][1]] = 1

        # Current opponent position (39 - 47)
        x[39 + (server_positions[0][0] * 3) + server_positions[0][1]] = 1
        # Previous opponent position (48 - 56)
        x[48 + (server_positions[1][0] * 3) + server_positions[1][1]] = 1
        
    # Player stroke (18 - 33)
    x[18 + shot_dict[current_shot]] = 1
    # Player stroke modifiers (34 - 35)
    if not cur_shot_ex == None:
        x[34 + cur_shot_ex] = 1
    # Removing any potential net-cord information for player shot (36)
    x[36] = 0
    # Player shot direction (36 - 38)
    x[36 + current_direction - 1] = 1

    # Opponent stroke (57 - 72)
    x[57 + shot_dict[previous_shot]] = 1
    # Opponent stroke modifiers (73 - 75)
    if not prev_shot_ex == None:
        x[73 + prev_shot_ex] = 1
    # Opponent shot direction (76 - 78)
    x[76 + prev_direction - 1] = 1

    # Rally length (up to players last shot) (79)
    x[79] = len(start_indices) - ended_early

    return x


# Removes any lets at the start of a rally string
# rally - the rally string for the current point
def remove_lets(rally):
    start_index = 0

    while rally[start_index] == "c":
        start_index += 1 

    return rally[start_index:]


def clean_surfaces(x):
    x = x.split(" ")
    
    if x[0][0] == "I":
        return x[1].lower()

    return x[0].lower()


# Updates the position of a player
# positions - the current and previous position of a player
# approach - True if a player is currently performing an approach shot, False otherwise
# col - the column of a players new court position
# depth - the row of a players new court position 
def update_position(positions, approach, col, depth):
    if (positions[1] == [0, 1]) and (positions[0] == [2, 2]):
                positions[0] = [1, 2]

    elif (positions[1] == [2, 1]) and (positions[0] == [0, 2]):
        positions[0] = [1, 2]

    else:
        if approach:
            positions[1] = [col, depth - 1]
            approach = False
        
        else:
            positions[1] = positions[0]

        positions[0] = [col, depth]

    return approach