.PHONY: format lint test
format:
\truff check --fix .
\tblack .
\tisort .
lint:
\truff check .
\tblack --check .
\tisort --check-only .
test:
\tpytest -q
