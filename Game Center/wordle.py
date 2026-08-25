import pandas as pd
class Wordle:
    def __init__(self, data_path:str="datasets\\Anees_Curated_Word_Bank.csv"):
        self.Attempts = 6
        self.guess = ""
        self.data = pd.read_csv(data_path)["الكلمة"]

    def _checking_guess_length(self, guess):
        '''
        checks the input length from the user.
        '''
        return len(guess)
    
    def  _letter_in_word(self, letter:str, word:str) -> bool:
        '''
        checks if a letter in the word.
        '''
        if letter in word:
              return True
        else:
            return False

    def  _right_position(self,letter:str, word:str, guess:str) -> bool:
        '''
        checks if the letter exist in the right position.

        '''
        if self._letter_in_word(letter, word)== True and guess.find(letter) == word.find(letter):
              return True

    def  _wrong_position(self,letter:str, word:str, guess:str) -> bool:
        '''
        checks if the letter in the word but in the wrong position.

        '''
        if self._letter_in_word(letter, word)== True and guess.find(letter) != word.find(letter):
              return True
        
    def _bring_index(self, guess:str, letter:str, position):
        '''
        return the position of each letter found in the user answer

        '''
        index_of_the_word = guess.find(letter, position)
        return index_of_the_word


    def start(self):
        #starting the game now
        while self.Attempts > 0:
            guess = input("Enter Your Guess (input should be 5 letters long):").lower() #ensuring that everything inputed is lower case to avoid any miss counting from same letters
            
            if (self._checking_guess_length(guess) == 5) and guess.isalpha() and guess.isascii(): #checks if the input is 5 letter long and all alphabetic
                user_guess_list = [""] * 5 #creating a list of 5 empty slots to store the answer while the user did not guess the answer correctly from the first time
                position =0

                for letter in guess:
                    # Case A -> correct letter in the correct positon (uppercase)
                    if self._right_position(letter, self.word, guess) == True :
                        index = self._bring_index(guess, letter, position)
                        user_guess_list[index] = letter.upper()

                    # Case B -> correct letter in the wrong position (Asteriks)
                    elif self._wrong_position(letter, self.word, guess) == True:
                        index = self._bring_index(guess , letter, position)
                        user_guess_list[index] = "*"+ letter.lower()+"*"

                    # Case C -> letter does not exist in the word (Underscore)
                    else:
                        user_guess_list[position] = "_" + letter + "_"
                    
                    position += 1 #since .find() and .index() only return the first occurance of the letter, sometimes the word contain the same letter twice like in (attic), position used to track the position of each letter so repeated letter are counted too
                
                print(f"{user_guess_list}") #display the current guess list made
                self.Attempts -=1
                print(f"You Have {self.Attempts} Attempt(s) Left")
                
                if guess == self.word:
                    print(f"The Word of the Day is {guess} You Won! Congrats")
                    break
            else:
                print("Invalid input. Please enter exactly 5 English alphabetic letters.")

        if self.Attempts == 0:
            print(f"Game Over, You Lost")
            print(f"The Word Of Today Is {self.word}")