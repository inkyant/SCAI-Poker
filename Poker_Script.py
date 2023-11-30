import csv, re, os

#PLACE SCRIPT IN THE SAME DIRECTORY AS PLURIBUS LOGS
fields = ["Cards", "Table Cards", "Move History", "Outcome"]

#Iterate through all files

for filename in os.listdir():
    if filename[-4:] != ".txt":
        continue
    f = open(filename, "r") 
    rows = []
    #Dictionary needs to be adjusted according to the players in the game
    players = {
        "Pluribus:": 0,
        "MrWhite:": 1,
        "Gogo:": 2,
        "Budd:": 3,
        "Eddie:": 4,
        "Bill:": 5
    }

    player_cards = ''
    table_cards = 'N/A'
    move_history = []
    for item in f:
        #collect card values from MrWhite
        if item[0:17] == "Dealt to Pluribus":
            player_cards = item[19:24]
        
        #Collect Moves
        #Format of Moves: (Name, Move)
        name = re.match('^[a-zA-Z]+:', item)
        if name:
            move = item[item.index(":"):]
            if re.match('folds', move):
                #TODO: implement moves as an array of numbers
                pass
                
            move_history.append((players[name.group()], item[item.index(" ")+1:item.index("\n")]))

        elif item[-4:-1] == "pot":
            win = ''
            #Check if MrWhite won
            if item[0:item.index(" ")] == "Pluribus":
                win = 1
            else:
                win = 0
        #Compile table cards
        elif item[0:5] == "Board":
            table_cards = item[7:item.index("]")]
        
        elif item == "\n":
            #The rest of this "if" branch shouldn't execute if no moves were made
            if len(move_history) == 0:
                continue
            #Make temporary move array
            move_array = []
            for element in move_history:
                move_array.append(element)
            rows.append([player_cards, table_cards, move_array, win])
            move_history.clear()
            table_cards = 'N/A'
        else:
            continue

    f.close()

#Write to CSV

with open('Poker_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #Write first row 
        writer.writerow(fields)

        #Loop through every row and write to csv
        for row in rows:
            writer.writerow(row)
    




