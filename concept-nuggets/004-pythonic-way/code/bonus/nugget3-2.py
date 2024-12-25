class Pasta:
    @staticmethod
    def boil_water(temperature):
        return f"Boiling water at {temperature} degrees."

    def __init__(self, type):
        self.type = type

# Using the static method without creating an instance
print(Pasta.boil_water(100))  # Boiling water at 100 degrees.