"""
LSTM Model Training for Player Transfer Value Prediction
Uses injury data and performance metrics to predict market values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AI TRANSFER IQ - LSTM MODEL TRAINING")
print("="*70)

# Load data
df = pd.read_csv('injury_with_market_values_cleaned.csv')
print(f"\n✓ Dataset loaded: {df.shape}")
print(f"✓ Columns: {len(df.columns)}")

# Select features for prediction
feature_cols = [
    'season_days_injured', 'total_days_injured', 
    'season_minutes_played', 'season_games_played', 
    'total_minutes_played', 'total_games_played',
    'age', 'height_cm', 'weight_kg', 'bmi',
    'work_rate_numeric', 'position_numeric',
    'cumulative_minutes_played', 'cumulative_games_played',
    'minutes_per_game_prev_seasons', 'avg_days_injured_prev_seasons',
    'avg_games_per_season_prev_seasons', 'significant_injury_prev_season',
    'cumulative_days_injured', 'season_days_injured_prev_season'
]

# Filter features that exist in the dataset
available_features = [col for col in feature_cols if col in df.columns]
print(f"\n✓ Using {len(available_features)} features")

# Target variable
target = 'market_value_eur'

# Prepare data
X = df[available_features].values
y = df[target].values

print(f"\n✓ Feature matrix shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Target range: {y.min():,.0f} to {y.max():,.0f} EUR")

# Remove any rows with invalid target values
valid_idx = y > 0
X = X[valid_idx]
y = y[valid_idx]
print(f"✓ Valid samples: {len(y)}")

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# Reshape for LSTM: [samples, timesteps, features]
# Using timesteps=1 for static features (can be extended for time-series)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"\n✓ LSTM input shape: {X_train_lstm.shape}")

# Build LSTM Model
print("\n" + "="*70)
print("BUILDING LSTM MODEL")
print("="*70)

model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, 
         input_shape=(1, len(available_features))),
    Dropout(0.3),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n✓ Model compiled successfully\n")
model.summary()

# Train model
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_lstm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\n✓ Training completed")

# Make predictions
print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

y_pred_train = model.predict(X_train_lstm, verbose=0).flatten()
y_pred_test = model.predict(X_test_lstm, verbose=0).flatten()

# Inverse transform to original scale
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_train_orig = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
train_r2 = r2_score(y_train_orig, y_pred_train_orig)
test_r2 = r2_score(y_test_orig, y_pred_test_orig)

print(f"\n📊 TRAINING METRICS:")
print(f"  - RMSE: €{train_rmse:,.2f}")
print(f"  - MAE:  €{train_mae:,.2f}")
print(f"  - R²:   {train_r2:.4f}")

print(f"\n📊 TEST METRICS:")
print(f"  - RMSE: €{test_rmse:,.2f}")
print(f"  - MAE:  €{test_mae:,.2f}")
print(f"  - R²:   {test_r2:.4f}")

# Calculate MAPE
test_mape = np.mean(np.abs((y_test_orig - y_pred_test_orig) / y_test_orig)) * 100
print(f"  - MAPE: {test_mape:.2f}%")

# Save model
model.save('lstm_market_value_model.h5')
print("\n✓ Model saved: lstm_market_value_model.h5")

# Save scalers
import pickle
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(available_features, f)
print("✓ Scalers saved: scaler_X.pkl, scaler_y.pkl, feature_names.pkl")

# Visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(18, 5))

# Plot 1: Training History
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Model Training History', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual
plt.subplot(1, 3, 2)
plt.scatter(y_test_orig/1e6, y_pred_test_orig/1e6, alpha=0.5, s=20)
min_val = min(y_test_orig.min(), y_pred_test_orig.min()) / 1e6
max_val = max(y_test_orig.max(), y_pred_test_orig.max()) / 1e6
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Value (€ millions)', fontsize=12)
plt.ylabel('Predicted Value (€ millions)', fontsize=12)
plt.title(f'Predictions vs Actual\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: Sample Predictions
plt.subplot(1, 3, 3)
sample_size = min(50, len(y_test_orig))
indices = np.arange(sample_size)
plt.plot(indices, y_test_orig[:sample_size]/1e6, 'o-', 
         label='Actual', markersize=5, linewidth=2)
plt.plot(indices, y_pred_test_orig[:sample_size]/1e6, 's-', 
         label='Predicted', markersize=5, linewidth=2, alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Market Value (€ millions)', fontsize=12)
plt.title(f'First {sample_size} Test Predictions', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: lstm_training_results.png")

# Feature importance (approximate using permutation)
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

baseline_score = test_r2
importance_scores = []

for i, feature in enumerate(available_features):
    X_test_permuted = X_test_lstm.copy()
    np.random.shuffle(X_test_permuted[:, 0, i])
    y_pred_permuted = model.predict(X_test_permuted, verbose=0).flatten()
    y_pred_permuted_orig = scaler_y.inverse_transform(y_pred_permuted.reshape(-1, 1)).flatten()
    permuted_r2 = r2_score(y_test_orig, y_pred_permuted_orig)
    importance = baseline_score - permuted_r2
    importance_scores.append(importance)

feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print(f"\n📈 Final Test R² Score: {test_r2:.4f}")
print(f"📈 Final Test MAE: €{test_mae:,.2f}")
print(f"📈 Final Test MAPE: {test_mape:.2f}%")
print("\n🎯 Model ready for predictions!")
