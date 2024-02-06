# Head Related Transfer Functions (HRTFs)

The contents in this directory are for using downloading and examining 
hrtf datasets.

## The CIPIC Dataset

Right now, we only support scripts and processing of the 
[CIPIC Database](https://ieeexplore.ieee.org/document/969552). Within the 
`cipic/` directory, you will find a bash script that will download the subject
files from the hosting site, SOFA conventions. This site hosts many other
datasets that may be of interest to us in the future. Annoyingly, they 
store the datasets in this cumbersome Spatially Oriented Format for Acoustics 
(SOFA) file format. 

Because of this, the easiest way to interact with the dataset is by using 
the `python-sofa` package, which can be installed via:

```
python -m pip install python-sofa
```

See the `cipic_db.py` file for an example on how use the `sofa` package.
However, the functionality provided in the `cipic_db.py` file should remove the
need to interact with `sofa` at all. 

Currently the `download.sh` script in the `cipic/` directory only downloads 
three subject files:

* `subject_012.sofa`: SOFA file for a subject chosen arbitrarily from the subject
set

* `subject_021.sofa`: SOFA file for [KEMAR](https://www.grasacoustics.com/industries/audiology/kemar#:~:text=KEMAR%20%E2%80%93%20the%20manikin%20for%20hearing,%2C%20acoustic%20impedance%2C%20and%20modes.) mannequin with large pinnae.

* `subject_165.sofa`: SOFA file for [KEMAR](https://www.grasacoustics.com/industries/audiology/kemar#:~:text=KEMAR%20%E2%80%93%20the%20manikin%20for%20hearing,%2C%20acoustic%20impedance%2C%20and%20modes.) mannequin with small pinnae.

Their relevant sofa attributes can be extracted in python using there IDs as
integer keys to the dictionary maintained by the `CipicDatabase` object. Thus,
a possible way to get the Head Related Impulse Response (HRIR) from the 
mannequin with large pinnae could:

```
hrir  = CipicDatabase.subjects[165].getHRIRFromIndex(<INDEX>, <CHANNEL>)
```

The various python scripts in this directory can be examined as more involved
examples of interacting with the `CipicDatabase` object.
