import subprocess

command_list = [
    'python src/downloader.py',
    #'python src/create_features.py',
    #'python src/model.py'
]

for command in command_list:
    subprocess.run(command.split(' '))