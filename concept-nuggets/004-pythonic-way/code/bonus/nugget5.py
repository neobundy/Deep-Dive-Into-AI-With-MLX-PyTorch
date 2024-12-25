class EspressoMachine:
    def brew(self):
        return "Brewing a rich espresso."

class TeaPot:
    def brew(self):
        return "Steeping some fine tea."

def start_brewing(brewer):
    print(brewer.brew())

# Both objects can be used in the same way
espresso_machine = EspressoMachine()
tea_pot = TeaPot()

start_brewing(espresso_machine)  # Brewing a rich espresso.
start_brewing(tea_pot)  # Steeping some fine tea.