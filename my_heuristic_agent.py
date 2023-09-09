def my_heuristic_agent(obs, config):
    import random
    import numpy as np

    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid

    # Returns True if dropping piece in column results in game win
    def check_winning_move(grid, config, col, piece):
        next_grid = drop_piece(grid, col, piece, config)
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(next_grid[row,col:col+config.inarow])
                if window.count(piece) == config.inarow:
                    return True,next_grid
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(next_grid[row:row+config.inarow,col])
                if window.count(piece) == config.inarow:
                    return True,next_grid
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(next_grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if window.count(piece) == config.inarow:
                    return True,next_grid
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(next_grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if window.count(piece) == config.inarow:
                    return True,next_grid
        return False,next_grid
    
    def opponent_mark(obsmark):
        return 1 if obsmark==2 else 2
    def valid_moves(grid, config):
        return [col for col in range(config.columns) if grid[0][col] == 0]
    
    def get_win_move(grid, config, obsmark):
        for col in valid_moves(grid, config):
            iswin, next_grid = check_winning_move(grid, config, col, obsmark)
            if iswin:
                return col
        return None
    
    def opponent_has_win_move(grid, config, obsmark):
        return get_win_move(grid, config, opponent_mark(obsmark)) != None
        
    def count_win_moves(grid, config, obsmark):
        possible_moves = valid_moves(grid, config)
        res = 0
        for col in possible_moves:
            iswin, _ = check_winning_move(grid, config, col, obsmark)
            if (iswin):
                res = res + 1
        return res
    
    def choose_move(grid, config, obsmark):
        win_move = get_win_move(grid, config, obsmark)
        if win_move:
            return [win_move],3 #win
        opponent_win_move = get_win_move(grid, config, opponent_mark(obsmark))
        if opponent_win_move:
            #print('necessary')
            return [opponent_win_move],1 #necessary
        possible_moves = valid_moves(grid, config)
        random.shuffle(possible_moves)
        not_bad_moves = []
        for col in possible_moves:
            next_grid = drop_piece(grid, col, obsmark, config)
            if not opponent_has_win_move(next_grid, config, obsmark):
                not_bad_moves.append(col)
                if count_win_moves(next_grid, config, obsmark) == 2:
                    return [col],2 #will win next time
        if len(not_bad_moves)>0:
            return not_bad_moves,1 # random
        return (possible_moves,0) # will loose
    
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    obsmark = obs.mark
    moves,good_move = choose_move(grid, config, obsmark)
    #print((moves,good_move))
    if good_move == 2 or good_move == 3:
        return moves[0] # we win
    if good_move == 1:
        if len(moves) == 1:
            return moves[0]
        possible_moves=[]
        for col in moves:
            next_grid = drop_piece(grid, col, obsmark, config) # my move
            _,opponent_good_move = choose_move(next_grid, config, opponent_mark(obsmark))
            if opponent_good_move>1:
                continue
            if opponent_good_move == 0:
                return col
            possible_moves.append(col)
        if len(possible_moves)>0:
            return possible_moves[0]
        return moves[0]
    if good_move == 0:
        return moves[0] # we loose