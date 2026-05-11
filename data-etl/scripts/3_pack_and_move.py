import argparse
import os
import tarfile
import shutil
import glob

def pack_and_move(year):
    base_dir = "/content/local_staging"
    if not os.path.exists(base_dir):
        base_dir = "./local_staging"

    # THE FIX: Your exact Google Drive path
    drive_dir = "/content/drive/MyDrive/Data-sets/BigDataData"

    if not os.path.exists("/content/drive"):
        print("Google Drive not found at /content/drive. Using local mock directory...")
        drive_dir = "./drive_mock/Data-sets/BigDataData"

    os.makedirs(drive_dir, exist_ok=True)

    tar_filename = f"climate_tensors_{year}.tar.gz"
    tar_path = os.path.join(base_dir, tar_filename)

    print(f"Compressing contents of {base_dir} into {tar_path}...")
    files_to_pack = glob.glob(os.path.join(base_dir, "*.npy"))
    if not files_to_pack:
        print(f"No .npy files found in {base_dir} to pack. Skipping...")
        return

    with tarfile.open(tar_path, "w:gz") as tar:
        for file in files_to_pack:
            print(f"Adding {file} to archive...")
            tar.add(file, arcname=os.path.basename(file))

    dest_path = os.path.join(drive_dir, tar_filename)
    print(f"Transferring archive to {dest_path}...")
    shutil.move(tar_path, dest_path)

    print(f"Cleaning up {base_dir} for next cycle...")
    for file in glob.glob(os.path.join(base_dir, "*")):
        try:
            if os.path.isfile(file) or os.path.islink(file):
                os.unlink(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    pack_and_move(args.year)
