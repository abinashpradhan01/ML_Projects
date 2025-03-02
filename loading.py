import pickle

# Define a dummy CO2 model
class DummyCO2Model:
    def predict(self, X):
        return [100] * len(X)  # Always returns 100 as a placeholder

# Save the model correctly
with open("co2_model.pkl", "wb") as f:
    pickle.dump(DummyCO2Model(), f)

print("✅ New co2_model.pkl created successfully!")
