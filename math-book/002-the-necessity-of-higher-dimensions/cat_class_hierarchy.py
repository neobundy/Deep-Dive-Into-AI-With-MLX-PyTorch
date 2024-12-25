class Animal:
    """A foundational class for all animals."""
    pass

class Mammal(Animal):
    """Represents the broader category of mammals, inheriting from Animal."""
    pass

class Cat(Mammal):
    """A detailed blueprint for creating cats, derived from Mammals."""

    def __init__(self, name, color, whiskers, patterns):
        """Initializes a new instance of a Cat."""
        self.name = name
        self.color = color
        self.whiskers = whiskers
        self.patterns = patterns

    def meow(self):
        """Enables the cat to vocalize."""
        print(f"{self.name} says: Meow!")

    def wear(self, accessory):
        """Outfits the cat with a chosen accessory."""
        self.accessory = accessory
        print(f"{self.name} is now adorned with {accessory}")


a_cat = Cat("Garfield", "Orange", "Luxuriant", "Striped")
a_cat.meow()  # Garfield proclaims: Meow!
a_cat.wear("a dapper bow tie")  # Garfield is now elegantly sporting a dapper bow tie