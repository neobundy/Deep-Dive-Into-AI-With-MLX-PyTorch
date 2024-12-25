import unittest

def mix_ingredients(ingredient1, ingredient2):
    return f"A mixture of {ingredient1} and {ingredient2}."

# Defining a test case
class TestMixingIngredients(unittest.TestCase):
    def test_mix_ingredients(self):
        """Test that ingredients are mixed correctly."""
        self.assertEqual(mix_ingredients("flour", "water"), "A mixture of flour and water.")

# Running the tests if the script is executed directly
if __name__ == '__main__':
    unittest.main()