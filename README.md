# virtual-generalist
Modeling complex co-morbidities in the Synthea patient simulator.

Watch our [5-minute overview](https://youtu.be/HqB_thGSm1c).

## Contents of this repository

* `virtual_generalist_module/` virtual_generalist_ckd module and lookup tables
	
* `notebooks/` Databricks notebooks to run the feature engineering and model building code on simulated data. We include HTML versions as well, if you just want to read it or copy code.

* `doc/` Experiments, medical code taxonomies

* `docs/` HTML outputs visualized on [web pages](https://github.com/rmhorton/virtual-generalist/blob/main/docs/index.md)

* `paper/` text and citations for Synthea competition submission. 

* `R/` Reusable R code 


## Links to Synthea resources

* Github repo for the Synthea patient simulation system:
https://github.com/synthetichealth/synthea

	- The [Basic Setup and Running](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running) page explains how to obtain and run the jar file without building the whole system. It works fine on open java.


* This file contains the records of 1000 simulated people in CSV format:
https://storage.googleapis.com/synthea-public/synthea_sample_data_csv_apr2020.zip

* This 1000 patient dataset is a sample from a bigger dataset which has records of 1M fake people: https://synthea.mitre.org/downloads.

## Attributes and concepts

* [attributes.json](https://github.com/rmhorton/virtual-generalist/blob/main/doc/attributes.json)
	- This file was generated with the command `./gradlew attributes`
	- Attributes read or written by either the framework or modules are given in this file.

* [concepts.csv](https://github.com/rmhorton/virtual-generalist/blob/main/doc/concepts.csv)
	- This file was generated with the command `./gradlew concepts`
	- All SNOMED-CT, RxNorm, and LOINC terms emitted by the simulation (including its modules) are included here (plus a few other things).

* [code_description_tally.csv](https://github.com/rmhorton/virtual-generalist/blob/main/doc/code_description_tally.csv)
	- Generated from a simulated population of 100k patients: `select code, description, count(*) tally from conditions group by code, description order by tally desc`
	- These were grouped by both code and description because there are some discrepancies (see below).

### SNOMED code-description descrepancies

Several codes have more than one description; these are the discrepancies that showed up in our 100k sample:
```
       code                   description tally
1 233604007          Pneumonia (disorder) 16326
2 233604007                     Pneumonia   496
3 427089005              Male Infertility    13
4 427089005 Diabetes from Cystic Fibrosis     9
5  55680006                 Drug overdose  4607
6  55680006      Drug overdose (disorder)     1
7  84757009                      Epilepsy  2086
8  84757009           Epilepsy (disorder)   191
```
