# virtual-generalist
Modeling complex co-morbidities in the Synthea patient simulator.

Watch our [5-minute overview](https://youtu.be/HqB_thGSm1c).

## Contents of this repository

* `virtual_generalist_module/` virtual_generalist_ckd module and lookup tables
	
* `notebooks/` Databricks notebooks to run the feature engineering and model building code on simulated data. We include HTML versions as well, if you just want to read it or copy code.

* `exp/` Experiments, examples, 

* `system_data/` data tables characterizing the current collection of Synthea modules, mapping ICD10 to SNOMED, etc.

* `docs/` HTML outputs visualized on [web pages](https://github.com/rmhorton/virtual-generalist/blob/main/docs/index.md)



## Links to Synthea resources

* Github repo for the Synthea patient simulation system:
https://github.com/synthetichealth/synthea

	- The [Basic Setup and Running](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running) page explains how to obtain and run the jar file without building the whole system. It works fine on open java.


* This file contains the records of 1000 simulated people in CSV format:
https://storage.googleapis.com/synthea-public/synthea_sample_data_csv_apr2020.zip

* This 1000 patient dataset is a sample from a bigger dataset which has records of 1M fake people: https://synthea.mitre.org/downloads.
