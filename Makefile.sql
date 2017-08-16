.PHONY: auto build test docs clean

test:
	rm -f .coverage
	. local/bin/activate && nosetests

