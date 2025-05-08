install-build:
	uv pip install build setuptools

build-clean:
	rm -rf dist/*

build: install-build build-clean
	python setup.py sdist bdist_wheel

publish: build
	twine upload dist/*
