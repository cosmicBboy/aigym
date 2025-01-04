install-build:
	pip install build

build-clean:
	rm -rf dist/*

build: install-build build-clean
	python setup.py sdist bdist_wheel

publish: build
	twine upload dist/*
