


card_dict = {"a": 1, "2": 2, "3": 3, "4": 4, "5": 5,
             "6": 6, "7": 7, "8": 8, "9": 9, "t": 10,
             "j": 11, "q": 12, "k": 13, "ace": 1, "10": 10, "jack": 11, "queen": 12, "king": 13}
suit_dict = {"h": 1, "d": 2, "c": 3, "s": 4, "hearts": 1, "diamonds": 2, "clubs": 3, "spades": 4}

reverse_card_dict = {
    1: "ace",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "jack",
    12: "queen",
    13: "king"
}

reverse_suits_dict = {
    1: "hearts",
    2: "diamonds",
    3: "clubs",
    4: "spades"
}

def query(pred_func):
    first = True
    while first or input().lower() == 'yes':
        first = False
        cards = []

        while len(cards) < 14:
            
            noCard = True
            while noCard:
                try:
                    print('give me a card')
                    card = card_dict[input()]
                    print('give me a suit')
                    suit = suit_dict[input()]
                    noCard = False
                except Exception:
                    print('Not a card, please try again\n') 

            print('\n')
            cards.append(card)
            cards.append(suit)


        print("your hand cards are: ")
        for i in range(2):
            print(reverse_card_dict[cards[i*2]] + " of " + reverse_suits_dict[cards[i*2 + 1]])

        print("\nyour table cards are: ")
        for i in range(2, 7):
            print(reverse_card_dict[cards[i*2]] + " of " + reverse_suits_dict[cards[i*2 + 1]])

        pred = pred_func(cards)

        print("\nYour chance of winning is: %.2f" %pred)

        print('would you like to continue? [yes]')