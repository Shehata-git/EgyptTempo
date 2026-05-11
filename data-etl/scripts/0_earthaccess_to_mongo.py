import os
import gc
import glob
import argparse
import earthaccess
import numpy as np
from pyhdf.SD import SD, SDC
from pymongo import MongoClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    # Connect to local Mongo
    client = MongoClient("mongodb://localhost:27017/")
    db = client['earthaccess_db']
    collection = db['temperature_data']

    print(f"Authenticating with Earthdata for year {args.year}...")
    earthaccess.login(strategy="environment")

    print(f"Searching for MOD11C1 data for {args.year}...")
    results = earthaccess.search_data(
        short_name="MOD11C1",
        bounding_box=(24.0, 22.0, 37.0, 32.0),
        temporal=(f"{args.year}-01-01", f"{args.year}-12-31")
    )

    if not results:
        print(f"No granules found for {args.year}.")
        return

    print(f"Downloading {len(results)} granules for {args.year}...")
    earthaccess.download(results, "./climate_data")

    print(f"Parsing HDF4 files for {args.year} and pushing to MongoDB...")
    downloaded_files = sorted(glob.glob("./climate_data/*.hdf"))

    # Egypt crop indices
    ROW_START, ROW_END = 1160, 1360
    COL_START, COL_END = 4080, 4340

    for file in downloaded_files:
        try:
            hdf = SD(file, SDC.READ)
            lst_obj = hdf.select('LST_Day_CMG')
            global_grid = lst_obj[:].astype(np.float32)

            temp_grid = global_grid[ROW_START:ROW_END, COL_START:COL_END]

            temp_grid[temp_grid == 0] = np.nan
            temp_grid = temp_grid * 0.02
            temp_grid = np.nan_to_num(temp_grid, nan=np.nanmean(temp_grid))

            filename = os.path.basename(file)
            date_str = filename.split('.')[1]

            # Upsert logic to avoid duplicates if a year fails and restarts
            document = {
                "date": date_str,
                "temperature_grid": temp_grid.tolist()
            }
            collection.update_one({"date": date_str}, {"$set": document}, upsert=True)

            hdf.end()
            os.remove(file)  # Delete immediately to save disk space
            gc.collect()

        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"Year {args.year} ingestion complete.")

if __name__ == "__main__":
    main()
