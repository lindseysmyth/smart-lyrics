
install: venv
	. venv/bin/activate && pip install -r requirements.txt

venv:
	test -d venv || python3 -m venv venv

run:
	@python3 app.py

clean:
	rm -rf venv
