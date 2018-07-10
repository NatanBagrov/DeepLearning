PYTHON_SCRIPTS = *.py utils/*.py models/*.py
TEXT_FILES = dependencies.txt README.txt
SANITY_FILES = run_sanity_check.py shrader_public.py sanity/*_input/*

submission_with_sanity.zip: $(PYTHON_SCRIPTS) $(TEXT_FILES) $(SANITY_FILES)
	zip submission_with_sanity.zip $(PYTHON_SCRIPTS) $(TEXT_FILES) $(SANITY_FILES)

upload_and_check_rishon: submission_with_sanity.zip
	ssh rishon "rm -rf for_sanity_check; mkdir for_sanity_check"
	scp submission_with_sanity.zip rishon:sanity/
	ssh rishon "cd rishon; unzip submission_with_sanity.zip; sbatch ./bash/run_python_job.bash run_sanity_check.py"

check_locally: submission_with_sanity.zip
	rm -rf for_sanity_check
	mkdir for_sanity_check
	cp submission_with_sanity.zip for_sanity_check/
	cd for_sanity_check;\
	unzip submission_with_sanity.zip;\
	python run_sanity_check.py
