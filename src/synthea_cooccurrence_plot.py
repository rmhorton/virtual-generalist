use synthea_patients;

drop table if exists encounter_feature_long;

create table encounter_feature_long as
with
pe1 as (
  select enc.id encounter
      , floor(datediff(enc.start, pat.birthdate)/365.24) age
      , pat.race
      , pat.ethnicity
      , pat.gender
    from patients pat join encounters enc on enc.patient=pat.id
      where enc.encounterclass in ('inpatient', 'outpatient')
)
,
pe2 as (
  select encounter, 
          concat_ws('_', 'gender', gender) gender, 
          concat_ws('_', 'ethnicity', ethnicity) ethnicity, 
          concat_ws('_', 'race', race) race, 
        case -- approximately 'MeSH' ranges according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3825015/
          when age <  2 then 'age_00_01' -- we'll only keep adults
          when age <  5 then 'age_02_04' -- we'll only keep adults
		  when age < 12 then 'age_05_11' -- we'll only keep adults
          when age < 18 then 'age_12_17' -- we'll only keep adults
          when age < 24 then 'age_18_23'
          when age < 44 then 'age_24_43'
          when age < 65 then 'age_44_64'
          when age < 80 then 'age_65_79'
          when age >=80 then 'age_80_plus'
          else 'age_unknown'
        end age_group
    from pe1
    where age >= 18
)
,
code_tally as (
  select code, count(*) tally from conditions group by code
)
,
encounter_condition_long as (
  select e.id encounter, c.description condition
    from encounters e
    join conditions c on c.patient = e.patient
    join code_tally ct on ct.code = c.code
    join pe2 on e.id = pe2.encounter
    where ct.tally > 100
      and c.start < e.stop
      and (c.stop > e.stop or c.stop is null)
)
select encounter, condition as feature from encounter_condition_long
union
select encounter, stack(4, gender, ethnicity, race, age_group) as feature from pe2
;


synthea_basket_item_sdf = spark.sql("select encounter as basket, feature as item from synthea_patients.encounter_feature_long")
# filter to just target conditions by joining to default.icd2synthea_map

synthea_ips = get_item_pair_stats(synthea_basket_item_sdf, item_col='item', min_count=100000)
synthea_ips.createOrReplaceTempView('synthea_ips')  # save for use in other plots

synthea_ips_bh = benjamini_hochberg_filter(synthea_ips, alpha=0.001, filter=True)
synthea_ips_nodes_pdf, synthea_ips_edges_pdf = get_nodes_and_edges_from_item_pair_stats(synthea_ips_bh.toPandas())
export_to_vis_js(synthea_ips_nodes_pdf, synthea_ips_edges_pdf, "Synthea co-occurrence plot", "synthea_cooccurrence.html")
