# virtual-generalist
A complex co-morbidities module for the Synthea patient simulator.

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
