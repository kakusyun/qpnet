import os


def get_corresponding_file(source_file, dest_path, dest_ext):
    _, ext_name = os.path.splitext(source_file)
    _, file_name = os.path.split(source_file)
    dest_basename = file_name.replace(file_name[-len(ext_name):], dest_ext)
    dest_name = os.path.join(dest_path, dest_basename)
    return dest_name
