install-build:
	pip install build

build: install-build
	python -m build
