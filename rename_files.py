import os

def fix_filenames(root_dir):
    """
    Fixes filenames in the specified directory by removing trailing dots. This is
    specifically a problem when working on a Windows machine because the filesytem
    is unable to recognize files with trailing dots, which can lead to issues when 
    trying to read or process the files.

    This will not be needed since the dataset included in the repo has already been fixed, 
    but this is included for reference if you want to work with the raw dataset found at
    https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
    """
    root_dir = os.path.abspath(root_dir)

    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith('.'):
                old_path = os.path.join(root, name)
                
                # Use Windows extended path
                old_path_ext = r"\\?\{}".format(old_path)
                
                new_name = name.rstrip('.')
                new_path = os.path.join(root, new_name)
                new_path_ext = r"\\?\{}".format(new_path)

                try:
                    if not os.path.exists(new_path_ext):
                        os.rename(old_path_ext, new_path_ext)
                        print(f"Renamed: {old_path} -> {new_path}")
                    else:
                        print(f"Skipped (exists): {new_path}")
                except Exception as e:
                    print(f"Error: {old_path} -> {e}")

fix_filenames("maildir")
