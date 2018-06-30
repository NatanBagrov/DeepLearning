#!/bin/bash

sbatch ./bash/run_python_job.bash ./models/solver_with_comparator.py 4 document
sbatch ./bash/run_python_job.bash ./models/solver_with_comparator.py 4 image
sbatch ./bash/run_python_job.bash ./models/solver_with_comparator.py 5 image
sbatch ./bash/run_python_job.bash ./models/solver_with_comparator.py 5 document
sbatch ./bash/run_python_job.bash ./models/solver_with_comparator.py 2 document
sbatch ./bash/run_python_job.bash ./models/solver_with_comparator.py 2 image