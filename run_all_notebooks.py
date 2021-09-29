import sys
import os

LOCAL_RUNNER_FILENAME = "run_all_notebooks.py"

# save command arguments
directories_where_execute = sys.argv[1:]

# if no argument was passed, default to all
if not directories_where_execute:
    print("INFO: No directory to execute specified, using all directories by default.")
    directories_where_execute = ['DFT', 'DFT + IQ',  'IQ', 'MP', 'MP + IQ', 'IQ - Data Augmentation']

for directory in directories_where_execute:

    # if selected dir doesn't exist
    if not os.path.isdir(directory):
        print("WARNING: " + directory + " does not exist, or is not a directory. Skipping.")
        continue

    # if runner doesn't exist in current directory
    if not os.path.isfile(directory + '/' + LOCAL_RUNNER_FILENAME):
        print("WARNING: " + directory + '/' + LOCAL_RUNNER_FILENAME +
              " does not exist, or is not a file. Notebooks in '" + directory + "' will not be executed.")
        continue

    # cd in directory containing the runner
    os.chdir(directory)

    # exec runner for current directory
    exec(open(LOCAL_RUNNER_FILENAME).read())

    # go back to root directory for the next iteration
    os.chdir('..')
