# virtual-generalist
Modeling complex co-morbidities in the Synthea patient simulator.

This project won second place in the 2021 [Synthetic Health Data Challenge](https://www.healthit.gov/topic/scientific-initiatives/pcor/synthetic-health-data-generation-accelerate-patient-centered-outcomes), sponsored by the HHS Office of the National Coordinator for Health Information Technology. Prize money will be donated to [sustainable Harvest International](https://www.sustainableharvest.org/).

This work is described in the [final report](https://github.com/rmhorton/virtual-generalist/blob/main/Virtual_Generalist_report.pdf) and the validation approach is described in the [appendixn](https://github.com/rmhorton/virtual-generalist/blob/main/validation/Virtual%20Generalist%20Validation%20Appendix.pdf).
We also have a [5-minute overview](https://youtu.be/HqB_thGSm1c) video.

## Contents of this repository

* [docs](docs): HTML outputs visualized on [web pages](https://rmhorton.github.io/virtual-generalist/)

* [exp](exp): Experiments, examples, etc.
	
* [notebooks](notebooks): Databricks notebooks to run the feature engineering and model building code on simulated data. We include HTML versions as well, if you just want to read it or copy code.

* [system_data](system_data): data tables characterizing the current collection of Synthea modules, mapping ICD10 to SNOMED, etc.

* [validation](validation): quantifying distribution differences between real and simulated populations.

* [virtual_generalist_module](virtual_generalist_module): CKD stage-trasition module and lookup tables




## Links to Synthea resources

* Github repo for the Synthea patient simulation system:
https://github.com/synthetichealth/synthea

	- The [Basic Setup and Running](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running) page explains how to obtain and run the jar file without building the whole system. It works fine on open java.


* This file contains the records of 1000 simulated people in CSV format:
https://storage.googleapis.com/synthea-public/synthea_sample_data_csv_apr2020.zip

* This 1000 patient dataset is a sample from a bigger dataset which has records of 1M fake people: https://synthea.mitre.org/downloads.
