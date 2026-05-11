import argparse
from pymongo import MongoClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    client = MongoClient("mongodb://localhost:27017/")
    db = client['earthaccess_db']

    # Safely delete only the specific year's data
    result = db.temperature_data.delete_many({"date": {"$regex": f"^A{args.year}"}})
    print(f"Wiped {result.deleted_count} NASA records for {args.year} from MongoDB to free up RAM.")

if __name__ == "__main__":
    main()
