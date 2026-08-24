import json
import random
from pathlib import Path
from collections import Counter


class HangmanGame:
    """Arabic word-guessing (hangman-style) game for children, part of Anees."""

    # Unicode isolate marks: force the whole revealed-word line to render as one
    # right-to-left run, so mixing Arabic letters with neutral "_" placeholders
    # doesn't get visually reordered by the browser/terminal's bidi algorithm.
    RLI = "⁧"
    PDI = "⁩"

    STAGES = [
        """
  -----
  |   |
      |
      |
      |
      |
---------
""",
        """
  -----
  |   |
  O   |
      |
      |
      |
---------
""",
        """
  -----
  |   |
  O   |
  |   |
      |
      |
---------
""",
        """
  -----
  |   |
  O   |
 /|   |
      |
      |
---------
""",
        """
  -----
  |   |
  O   |
 /|\\  |
      |
      |
---------
""",
        """
  -----
  |   |
  O   |
 /|\\  |
 /    |
      |
---------
""",
        """
  -----
  |   |
  O   |
 /|\\  |
 / \\  |
      |
---------
""",
    ]

    def __init__(self, data_path: str = None):
        """Loads the word list (word -> hint) from a JSON file next to this module."""
        if data_path is None:
            data_path = Path(__file__).parent / "hangman_words.json"
        with open(data_path, "r", encoding="utf-8") as f:
            self.wordlist = json.load(f)

    def start(self):
        """Starts the game loop: picks a random word and handles guesses until win/lose."""
        word, hint = random.choice(list(self.wordlist.items()))
        print("خمّن الكلمة!")

        letter_guessed = ""
        tried_letters = set()
        wrong_guesses = 0
        max_chances = len(self.STAGES) - 1

        try:
            while wrong_guesses < max_chances:
                print()
                print(self.display_word(word, letter_guessed))
                guess = input("أدخل حرفًا: ")

                error = self.validate_input(guess, tried_letters)
                if error:
                    print(error)
                    continue

                guess = guess.strip()
                tried_letters.add(guess)
                if guess in word:
                    letter_guessed += guess * word.count(guess)
                else:
                    wrong_guesses += 1
                    print(self.STAGES[wrong_guesses])
                    if wrong_guesses == max_chances - 1:
                        print(f"\nتلميح: {hint}")

                if self.check_win(letter_guessed, word):
                    print("\nتهانينا! لقد خمّنت الكلمة:", word)
                    break

            if wrong_guesses == max_chances:
                print("\nللأسف، لقد خسرت! كانت الكلمة:", word)

        except KeyboardInterrupt:
            print("\nتم إنهاء اللعبة.")

    def display_word(self, word: str, letter_guessed: str) -> str:
        """Returns the word with underscores for unguessed letters, wrapped in
        RTL isolate marks so the mix of Arabic letters and "_" displays in the
        correct visual order."""
        revealed = " ".join(char if char in letter_guessed else "_" for char in word)
        return f"{self.RLI}{revealed}{self.PDI}"

    def check_win(self, letter_guessed: str, word: str) -> bool:
        """Checks if every letter of the word has been guessed (right counts included)."""
        return Counter(letter_guessed) == Counter(word)

    def validate_input(self, guess: str, tried_letters: set) -> str:
        """Returns an error message if the guess is invalid, or '' if it's valid.
        tried_letters holds every letter attempted so far (right AND wrong), so a
        repeated wrong guess is rejected instead of silently burning another life."""
        guess = guess.strip()
        if len(guess) == 0:
            return "يجب إدخال حرف واحد!"
        if len(guess) != 1:
            return "يجب إدخال حرف واحد فقط!"
        if not (guess.isalpha() and ("؀" <= guess <= "ۿ")):
            return "يجب إدخال حرف عربي فقط!"
        if guess in tried_letters:
            return "لقد خمّنت هذا الحرف من قبل!"
        return ""
