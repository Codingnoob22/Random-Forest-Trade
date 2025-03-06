# Fixed Data Preparation & Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data

df = pd.read_csv('BTC_1min.csv')

    
'''
# Fixed Feature Engineering with proper returns handling
''' 
def technical_features(df, window_sizes=[3, 5, 8]):
    
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    '''
    Technical Indicators features
    '''
    #Moving Average(3, 5, 8)
    for window in window_sizes:
        # Simple Moving Average
        df_copy[f'SMA_{window}'] = df_copy['Close'].rolling(window).mean()
                
        # Relative Strength Index (RSI)
        gain = df_copy['Close'].where(df_copy['Close'] > 0, 0)
        loss = -df_copy['Close'].where(df_copy['Close'] < 0, 0)

        # Calculate first RSI value
        avg_gain = pd.Series([np.nan] * 13 + [gain.iloc[1:8].mean()], index=gain.index[:14])._append(gain.iloc[14:])
        avg_loss = pd.Series([np.nan] * 13 + [loss.iloc[1:8].mean()], index=loss.index[:14])._append(loss.iloc[14:])
        
        # Use ewm for subsequent values
        avg_gain = avg_gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = avg_loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))


    # MACD(5, 13)
    ema5  = df_copy['Close'].ewm(span=5, adjust=False).mean()
    ema13 = df_copy['Close'].ewm(span=13, adjust=False).mean()
    df_copy['MACD'] = ema5 - ema13
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=4, adjust=False).mean()
    df_copy['MACD_Hist']   = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # Stochastic(7)
    n = 7
    low_min  = df_copy['Low'].rolling(window=n).min()
    high_max = df_copy['High'].rolling(window=n).max()
    
    # ATR Calculation using shorter period mixing with Keltner
    tr1  = df_copy['High']-df_copy['Low']
    tr2  = abs(df_copy['High'] - df_copy['Close'].shift())
    tr3  = abs(df_copy['Low'] - df_copy['Close'].shift())
    tr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr5 = tr.ewm(span=5, adjust=False).mean()
    
    df_copy['Keltner_Middle'] = df_copy['Close'].ewm(span=10, adjust=False).mean()
    df_copy['Keltner_Upper']  = df_copy['Keltner_Middle'] + (atr5 * 1.5)
    df_copy['Keltner_Lower']  = df_copy['Keltner_Middle'] - (atr5 * 1.5)
    
    # Momentum, using 3 period momentum
    df_copy['Momentum'] = df_copy['Close'] - df_copy['Close'].shift(3)
    
    # Volume with Moving Average, shorter period
    '''
    RoC = Rate of Change
    Sma = Shorter(Period) MA
    '''
    df_copy['Vol_RoC']   = df_copy['Volume'].pct_change(3) * 100
    df_copy['Vol_Sma']   = df_copy['Volume'].rolling(window=5).mean()
    df_copy['Vol_Ratio'] = df_copy['Volume']/df_copy['Vol_Sma']
    
    # Clean data
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df_copy

'''
Building the feature to make the training better 
''' 
# Preprocess data
df['Returns'] = df['Close'].pct_change()

# Create target variable
df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)

# Apply feature engineering and CAPTURE THE RETURNED DATAFRAME
df = technical_features(df)


features = [col for col in df.columns if col not in [
    'Timestamp', 'Target', 'Returns', 'Open', 'High', 'Low', 'Close', 'Volume'
]]
target = 'Target'


# Remove highly correlated features to reduce dimensionality
def remove_correlated_features(df, features, threshold=0.95):
    """Remove highly correlated features to reduce overfitting"""
    features_df = df[features]
    corr_matrix = features_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features")
    
    return [f for f in features if f not in to_drop]

# Apply correlation filter
features = remove_correlated_features(df, features, threshold=0.95)
print(f"Final feature count: {len(features)}")

target = 'Target'

'''
#XCOxDeepSeekAIxClaude3.7
Training engineering 
'''

'''
Model training library 
'''
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# Time-based split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]
# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])
y_train = train[target]
y_test = test[target]

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


