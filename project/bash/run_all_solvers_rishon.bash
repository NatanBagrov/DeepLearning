#!/bin/bash

sbatch ./bash/run_python_job.bash ./solvers/solver_greedy.py 4 document
sbatch ./bash/run_python_job.bash ./solvers/solver_greedy.py 4 image
sbatch ./bash/run_python_job.bash ./solvers/solver_greedy.py 5 image
sbatch ./bash/run_python_job.bash ./solvers/solver_greedy.py 5 document
sbatch ./bash/run_python_job.bash ./solvers/solver_greedy.py 2 document
sbatch ./bash/run_python_job.bash ./solvers/solver_greedy.py 2 image