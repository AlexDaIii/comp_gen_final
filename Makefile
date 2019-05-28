dist:
	zip -r final.zip README.md ./models/*.py ./models/*.sh ./output ./data/expr_labels.csv ./data/features.csv ./data/labels.csv ./preprocess *.py *.sh

clean:
	rm -rf *.tar.gz output/*