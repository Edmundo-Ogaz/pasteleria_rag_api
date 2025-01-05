env:
	python3 -m venv venv

activate:
	source venv/bin/activate

deactivate:
	deactivate

run:
	python ./app.py