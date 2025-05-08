install-build:
	uv pip install build setuptools

build-clean:
	rm -rf dist

build: install-build build-clean
	python -m build

publish: build
	twine upload dist/*
