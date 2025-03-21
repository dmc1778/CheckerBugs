import json
from collections import Counter
import re
from datetime import datetime
from statistics import mean
import sys
sys.path.insert(0, '/home/USER/repository/TensorGuard')
from utils.utils import load_json

def general_overview(data):
    total_commits = len(data)
    unique_libraries = set(entry['LibraryName'] for entry in data)
    return total_commits, unique_libraries

# 2. Commit Information
def commit_info(data):
    commit_dates = [datetime.fromisoformat(entry['date']) for entry in data]
    earliest_commit = min(commit_dates)
    latest_commit = max(commit_dates)
    commit_frequency = Counter([date.month for date in commit_dates])  # Commit frequency by month
    return earliest_commit, latest_commit, commit_frequency

# 3. Changes Analysis
def changes_analysis(data):
    file_changes = Counter()
    total_patches = 0
    for entry in data:
        for change in entry['changes']:
            file_changes[change['name']] += 1
            total_patches += len(change['patches'])
    
    return file_changes, total_patches

# 4. Patch Details
def patch_details(data):
    patch_sizes = []
    for entry in data:
        for change in entry['changes']:
            for patch in change['patches']:
                patch_size = patch['old_length']
                patch_sizes.append(patch_size)
    avg_patch_size = mean(patch_sizes) if patch_sizes else 0
    return avg_patch_size, patch_sizes

# 5. Commit Message Length
def commit_message_length(data):
    message_lengths = [len(entry['message']) for entry in data]
    avg_message_length = mean(message_lengths) if message_lengths else 0
    return avg_message_length


if __name__ == '__main__':
    lib_name = 'tensorflow'
    data_path = f"data/taxonomy_data/{lib_name}_test_data.json"
    data = load_json(data_path)
    
    total_commits, unique_libraries = general_overview(data)
    earliest_commit, latest_commit, commit_frequency = commit_info(data)
    file_changes, total_patches = changes_analysis(data)
    avg_patch_size, patch_sizes = patch_details(data)
    avg_message_length = commit_message_length(data)

print(f"Total Commits: {total_commits}")
print(f"Unique Libraries: {unique_libraries}")
print(f"Earliest Commit: {earliest_commit}")
print(f"Latest Commit: {latest_commit}")
print(f"Total Patches: {total_patches}")
print(f"Average Patch Size: {avg_patch_size}")
print(f"Average Commit Message Length: {avg_message_length}")