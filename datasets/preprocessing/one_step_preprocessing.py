import os

os.system('python extract_labelme_json.py')
os.system('python generate_json_and_gt.py')
os.system('python relabeling_location.py')
