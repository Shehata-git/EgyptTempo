import argparse
import os
import gc
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

def fetch_and_scale(year, mongo_uri, db_name, collection_name):
    print(f"Connecting to MongoDB at {mongo_uri}...")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    print(f"Fetching data for year {year}...")
    # THE FIX: NASA date format is AYYYYDDD (e.g., A2014001)
    query = {"date": {"$regex": f"^A{year}"}}
    cursor = collection.find(query).sort("date", 1)

    unique_dates = sorted(collection.distinct("date", query))
    if not unique_dates:
        print(f"No data found for year {year}")
        return

    # Egypt crop dimensions
    ROW_START, ROW_END = 1160, 1360
    COL_START, COL_END = 4080, 4340
    ROWS = ROW_END - ROW_START
    COLS = COL_END - COL_START

    num_days = len(unique_dates)
    climate_grid = np.zeros((num_days, ROWS, COLS), dtype=np.float32)

    for i, doc in enumerate(cursor):
        grid = np.array(doc["temperature_grid"], dtype=np.float32)
        climate_grid[i] = grid

    print(f"Loaded {num_days} days of data. Shape: {climate_grid.shape}")

    original_shape = climate_grid.shape
    flat_data = climate_grid.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(flat_data)

    scaler_params = np.array([scaler.data_min_[0], scaler.data_max_[0]])

    out_dir = "/content/local_staging"
    if not os.path.exists(out_dir):
        out_dir = "./local_staging"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "scaler_params.npy"), scaler_params)

    scaled_grid = scaled_data.reshape(original_shape)

    save_path = os.path.join(out_dir, f"raw_scaled_{year}.npy")
    print(f"Saving scaled data to {save_path}...")
    np.save(save_path, scaled_grid)

    print("Cleaning up memory...")
    del climate_grid
    del flat_data
    del scaled_data
    del scaled_grid
    gc.collect()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--mongo_uri", type=str, default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    args = parser.parse_args()

    fetch_and_scale(args.year, args.mongo_uri, "earthaccess_db", "temperature_data")
