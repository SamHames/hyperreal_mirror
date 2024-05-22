# Download and prepare stackoverflow data for analysis
#
# Requirements: python, wget
#
# Assume you will need around ~200GiB of hard drive space to run.
#

set -e

# Download archive dumps
mkdir -p data
wget -c -N -P data/ \
	https://archive.org/download/stackexchange/stackoverflow.com-Users.7z
wget -c -N -P data/ \
	https://archive.org/download/stackexchange/stackoverflow.com-Comments.7z
wget -c -N -P data/ \
	https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
	

# Setup a python environment for this experiment
python -m venv stackoverflow_analysis
source stackoverflow_analysis/bin/activate

wget https://hyperreal.app/tarball -O hyperreal.tar.gz
pip install ./hyperreal.tar.gz[stackexchange] leather tablib[html]

hyperreal stackexchange-corpus replace-sites \
	data/stackoverflow.com-Posts.7z \
	data/stackoverflow.com-Comments.7z \
	data/stackoverflow.com-Users.7z \
	data/stackoverflow.db

hyperreal stackexchange-corpus index \
	--doc-batch-size 50000 \
	data/stackoverflow.db \
	data/stackoverflow_index.db

hyperreal model \
	--clusters 1024 \
	--include-field Post \
	--min-docs 290 \
	--random-seed 2023 \
	--tolerance 0.001 \
	--iterations 100 \
	--restart \
	data/stackoverflow_index.db

python visuals.py

# Launch the webserver, then navigate to localhost:8080 in your browser
hyperreal stackexchange-corpus serve \
	data/stackoverflow.db \
	data/stackoverflow_index.db


