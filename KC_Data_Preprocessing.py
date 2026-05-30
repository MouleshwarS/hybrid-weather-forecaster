import pandas as pd
import numpy as np

def prepare_konkan_data(file_path, feature_scaler, target_scaler, seq_len, target_variable):
    """
    Standard Preprocessing (Log1p Version): 
    - Drops 'latitude' and 'longitude' from feature channels to avoid coordinate bias.
    - Retains lat/lon metadata for axis labeling and spatial reshaping.
    - Applies log1p (ln(1+x)) transformation to 'tp' and 'ws'.
    - Scaled and Reshaped for U-Net spatial dimensions.
    """
    df = pd.read_csv(file_path)
    
    # Storing geographic metadata for plotting
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    
    # Spatial Dimensions
    height = df['latitude'].nunique()
    width = df['longitude'].nunique()
    pixels_per_step = height * width
    num_timesteps = len(df) // pixels_per_step
    
    # Identifying split years -> get the first index of the specific year
    val_raw_idx = df.loc[df['year'] == 2019].index[0]
    test_raw_idx = df.loc[df['year'] == 2021].index[0]
    
    # Retaining only the atmospheric variables for the model input
    feature_cols = [c for c in df.columns if c not in ['latitude', 'longitude', 'year']]
    df_final = df[feature_cols] 
    
    feature_names = df_final.columns.tolist()
    
    X_np = df_final.values.astype('float32')
    y_np = df[[target_variable]].values.astype('float32')
    
    num_channels = X_np.shape[1] 

    # Log Transformation
    if target_variable in ['tp', 'ws']:
        y_np = np.log1p(y_np)
        
    if 'tp' in feature_names:
        X_np[:, feature_names.index('tp')] = np.log1p(X_np[:, feature_names.index('tp')])
    if 'ws' in feature_names:
        X_np[:, feature_names.index('ws')] = np.log1p(X_np[:, feature_names.index('ws')])

    # Reshape to Spatial Grid (T, H, W, C)
    X_grid = X_np.reshape(num_timesteps, height, width, num_channels)
    y_grid = y_np.reshape(num_timesteps, height, width, 1)
    
    # Sequence Generation
    X_seq, y_seq = [], []
    for i in range(len(X_grid) - seq_len):
        X_seq.append(X_grid[i : i + seq_len])
        y_seq.append(y_grid[i + seq_len])
    
    X_all = np.array(X_seq)
    y_all = np.array(y_seq)
    
    val_split = (val_raw_idx // pixels_per_step) - seq_len
    test_split = (test_raw_idx // pixels_per_step) - seq_len
    
    X_train_raw, y_train_raw = X_all[:val_split], y_all[:val_split]
    X_val_raw, y_val_raw = X_all[val_split:test_split], y_all[val_split:test_split]
    X_test_raw, y_test_raw = X_all[test_split:], y_all[test_split:]

    # Scaling
    X_train = feature_scaler.transform(X_train_raw.reshape(-1, num_channels)).reshape(X_train_raw.shape)
    X_val = feature_scaler.transform(X_val_raw.reshape(-1, num_channels)).reshape(X_val_raw.shape)
    X_test = feature_scaler.transform(X_test_raw.reshape(-1, num_channels)).reshape(X_test_raw.shape)

    y_train = target_scaler.transform(y_train_raw.reshape(-1, 1)).reshape(y_train_raw.shape)
    y_val = target_scaler.transform(y_val_raw.reshape(-1, 1)).reshape(y_val_raw.shape)
    y_test = target_scaler.transform(y_test_raw.reshape(-1, 1)).reshape(y_test_raw.shape)

    return {
        'train': (X_train, y_train),
        'val':   (X_val, y_val),
        'test':  (X_test, y_test),
        'meta': {
            'height': height, 
            'width': width, 
            'channels': num_channels,
            'feature_names': feature_names,
            'lat_range': (lat_min, lat_max),
            'lon_range': (lon_min, lon_max)
        }
    }