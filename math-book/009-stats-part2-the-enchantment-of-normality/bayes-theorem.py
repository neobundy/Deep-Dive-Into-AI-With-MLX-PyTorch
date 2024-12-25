# Prior probabilities
P_DiseaseA = 0.01
P_NoDiseaseA = 0.99

# Likelihoods
P_Positive_DiseaseA = 0.95
P_Positive_NoDiseaseA = 0.05

# Total probability of testing positive
P_Positive = (P_Positive_DiseaseA * P_DiseaseA) + (P_Positive_NoDiseaseA * P_NoDiseaseA)

# Posterior probability of having the disease given a positive test result
P_DiseaseA_Positive = (P_Positive_DiseaseA * P_DiseaseA) / P_Positive

print(f"The probability of having Disease A given a positive test result is: {P_DiseaseA_Positive:.2%}")
# The probability of having Disease A given a positive test result is: 16.10%