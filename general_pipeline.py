import subprocess

command_list = [
    'python src/downloader.py',
    'python src/cleaner.py',
    'python src/storm_outages.py',
    'python src/meteorological_api.py',
    #'python src/create_features.py',
    #'python src/model.py'
]

for command in command_list:
    subprocess.run(command.split(' '))