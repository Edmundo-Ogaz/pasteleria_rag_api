env:
	python3 -m venv venv

activate:
	source venv/bin/activate

deactivate:
	deactivate

requirement:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt

run:
	python ./app.py

nohup:
	nohup python ./app.py

ps:
	ps -fe

kill:
	kill [PID]

test:
	PYTHONPATH=. pytest test/test_jina.py

test-metod:
	PYTHONPATH=. pytest test/test_jina.py::test_get_similarity_whith_scores