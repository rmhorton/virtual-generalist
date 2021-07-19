This directory contains a variety of data files that help us understand the Synthea system and the technologies we use in the Virtual Generalist.

## ICD10 to SNOMED map

* `icd2synthea_map.csv`
 	- this is a hand-curated map of ICD10 codes to the SNOMED concepts we use in the Virtual Generalist. These include the more common SNOMED concepts found in Syntha EMR data (see `code_description_tally.csv`), as well as a few additional concepts we are considering adding.
 	- ICD10 codes are treated as regular expression patterns; the pattern `E11` matches any more specific code that contains `E11`.
 	- There is one row per ICD10 code pattern. If more than one ICD10 code must be present at the same time to indicate a SNOMED concept, all of the required ICD10 codes must have the same `icd_code_set` value.
 	- Additional materials related to ICD10-SNOMED mapping are in the directory [icd_snomed_mapping_misc](icd_snomed_mapping_misc)


## Attributes and concepts

* [attributes.json](https://github.com/rmhorton/virtual-generalist/blob/main/system_data/attributes.json)
	- This file was generated with the command `./gradlew attributes`
	- Attributes read or written by either the framework or modules are given in this file.

* [concepts.csv](https://github.com/rmhorton/virtual-generalist/blob/main/system_data/concepts.csv)
	- This file was generated with the command `./gradlew concepts`
	- All SNOMED-CT, RxNorm, and LOINC terms emitted by the simulation (including its modules) are included here (plus a few other things).
	- This file is read by the Databricks notebook to annotate the 

## Count of codes and descriptions

* [code_description_tally.csv](https://github.com/rmhorton/virtual-generalist/blob/main/system_data/code_description_tally.csv)
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
