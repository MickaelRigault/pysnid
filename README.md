[![PyPI](https://img.shields.io/pypi/v/pysnid.svg?style=flat-square)](https://pypi.python.org/pypi/pysnid)


# pysnid
simple python wrapper to run SNID

# Installation

Assuming you already have snid installed [see here](https://people.lam.fr/blondin.stephane/software/snid/)

Simply `pip install pysnid`

# Templates

This version of pysnid additionally works with the new [Super-SNID](https://github.com/dkjmagill/QUB-SNID-Templates) + the [Gutiérrez](https://arxiv.org/abs/1709.02487) 
templates. See the [SNID](https://people.lam.fr/blondin.stephane/index.html) webpage for installation instructions. Moreover, these templates and the typeinfo file have been updated to recover the type (e.g. Ia) and subtype (e.g. 91bg) as type-subtype (Ia-91bg), for all the types/subtypes included at the template folder.

# Usage

If you want to fit a spectra stored under `filename=YOUR_DIR_PATH/spectra.ascii`
Then:
```python
import pysnid
snidres = pysnid.run_snid(filename)
```

`snidres` is a custom made object (`SNIDReader`) that contains useful tools to access and visualise SNID input and output.
the input data is stored as `snidres.data` (DataFrame) and the snid result table as `snidres.results`. 

for instance:
```python
snidres.results
```

<p align="left">
  <img src="images/snidresults.png" width="400" title="results">
</p>


To visualize the template matching do (here the best matches 1, 2 and 4 (only the fist is shown by default) :
```python
snidres.show(models=[1,2,4])
```
<p align="left">
  <img src="images/show_top3.png" width="550" title="results">
</p>


#### Some more details

`pysnid.run_snid()` creates by default a new `h5` file with the same path as the input file but for the extension replaced by `_snid.h5`. Use `pysnid.SNIDReader.from_filename(outfile)` to read this file.
