import shutil

# Deleting an non-empty folder
dir_path1 = r"./build"
dir_path2 = r"./dist"
shutil.rmtree(dir_path1, ignore_errors=True)
print("Deleted '%s' directory successfully" % dir_path1)
shutil.rmtree(dir_path2, ignore_errors=True)
print("Deleted '%s' directory successfully" % dir_path2)
