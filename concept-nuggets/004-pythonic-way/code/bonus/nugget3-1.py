class Soup:
    _seasoning = 'Salt'

    @classmethod
    def update_seasoning(cls, new_seasoning):
        cls._seasoning = new_seasoning

    def __init__(self, ingredient):
        self.ingredient = ingredient

    def describe(self):
        return f"This soup contains {self.ingredient} seasoned with {self._seasoning}."

# Updating the seasoning for all soups
Soup.update_seasoning('Pepper')

# Both instances reflect the updated class state
tomato_soup = Soup('Tomato')
mushroom_soup = Soup('Mushroom')
print(tomato_soup.describe())  # This soup contains Tomato seasoned with Pepper.
print(mushroom_soup.describe())  # This soup contains Mushroom seasoned with Pepper.