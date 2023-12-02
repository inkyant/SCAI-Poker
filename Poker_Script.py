import csv, re, os

#PLACE SCRIPT IN THE SAME DIRECTORY AS PLURIBUS LOGS
card_dict = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13}
suit_dict = {"h": 1, "d": 2, "c": 3, "s": 4}
fields = ["Cards", "Move History", "Outcome"]
rows = []
#Iterate through all files

for filename in os.listdir():
    if filename[-4:] != ".txt":
        continue
    f = open(filename, "r") 

    #Read all lines to extract names
    lines = f.readlines()
    
    names = [lines[i][8: lines[i].index("(") - 1] for i in range(2, 8)] #Uses the left parenthesis to stop at space right after the name
    names.remove("Pluribus")
    players = {
        "Pluribus:": 0,
        names[0] + ":": 1,
        names[1] + ":": 2,
        names[2] + ":": 3,
        names[3] + ":": 4,
        names[4] + ":": 5
    }

    player_cards = "N/A"
    move_history = []
    for item in lines:
        #collect card values from MrWhite
        if item[0:17] == "Dealt to Pluribus":
            card_1 = card_dict[item[19:20]]
            suit_1 = suit_dict[item[20:21]]
            card_2 = card_dict[item[22:23]]
            suit_2 = suit_dict[item[23:24]]

            player_cards = f'{card_1}, {suit_1}, {card_2}, {suit_2}'




        
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
            #Check if Pluribus won won
            if item[0:item.index(" ")] == "Pluribus":
                win = 1
            else:
                win = 0
        #Compile table cards
        elif item[0:5] == "Board":
            #Use difference in index numbers to find how many cards are on the table
            diff = item.index("]") - 7
            table_card1 = card_dict[item[7:8]]
            table_suit1 = suit_dict[item[8:9]]

            table_card2 = card_dict[item[10:11]]
            table_suit2 = suit_dict[item[11:12]]

            table_card3 = card_dict[item[13:14]]
            table_suit3 = suit_dict[item[14:15]]
            #Diff 8: 3 Cards
            #Diff 11: 4 Cards
            #Diff 14: 5 Cards
            if diff == 8:
                player_cards += f', {table_card1}, {table_suit1}, {table_card2}, {table_suit2}, {table_card3}, {table_suit3}'
            elif diff == 11:
                table_card4 = card_dict[item[16:17]]
                table_suit4 = suit_dict[item[17:18]]
                player_cards += f', {table_card1}, {table_suit1}, {table_card2}, {table_suit2}, {table_card3}, {table_suit3}, {table_card4}, {table_suit4}'
            
            else:
                table_card4 = card_dict[item[16:17]]
                table_suit4 = suit_dict[item[17:18]]
                
                table_card5 = card_dict[item[19:20]]
                table_suit5 = suit_dict[item[20:21]]
                player_cards += f', {table_card1}, {table_suit1}, {table_card2}, {table_suit2}, {table_card3}, {table_suit3}, {table_card4}, {table_suit4}, {table_card5}, {table_suit5}'
        
        elif item == "\n":
            #The rest of this "if" branch shouldn't execute if no moves were made
            if len(move_history) == 0:
                continue
            #Make temporary move array
            move_array = []
            for element in move_history:
                move_array.append(element)
            rows.append([player_cards, move_array, win])
            move_history.clear()
            player_cards = 'N/A'
            
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
    




