import sys
import os

LOCAL_RUNNER_FILENAME = "run_all_notebooks.py"

# save command arguments
dirs = ['rotate', 'noise', 'mixed', 'flip']


for directory in dirs:

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
