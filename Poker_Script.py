import csv

#PLACE SCRIPT IN THE SAME DIRECTORY AS PLURIBUS LOGS
fields = ["Cards", "Table Cards", "Move History", "Outcome"]

f = open("pluribus_30.txt", "r")
rows = []

player_cards = ''
table_cards = 'N/A'
move_history = []
for item in f:
    #collect card values from MrWhite
    if item[0:16] == "Dealt to MrWhite":
        player_cards = item[18:23]
    #Collect Moves
    elif item[0:8] == "MrWhite:":
        move_history.append(item[item.index(" ")+1:item.index("\n")])
    elif item[-4:-1] == "pot":
        win = ''
        #Check if MrWhite won
        if item[0:item.index(" ")] == "MrWhite":
            win = 1
        else:
            win = 0
    #Compile table cards
    elif item[0:5] == "Board":
        table_cards = item[7:item.index("]")]
    
    elif item == "\n":
        #Make sure none of this executes if no moves were made
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
    




