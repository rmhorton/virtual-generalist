# table `encounter_feature_long` is created by SQL code in `synthea_cooccurrence_plot.sql`

synthea_basket_item_sdf = spark.sql("select encounter as basket, feature as item from synthea_patients.encounter_feature_long")
# filter to just target conditions by joining to default.icd2synthea_map

synthea_ips = get_item_pair_stats(synthea_basket_item_sdf, item_col='item', min_count=100000)
synthea_ips.createOrReplaceTempView('synthea_ips')  # save for use in other plots

synthea_ips_bh = benjamini_hochberg_filter(synthea_ips, alpha=0.001, filter=True)
synthea_ips_nodes_pdf, synthea_ips_edges_pdf = get_nodes_and_edges_from_item_pair_stats(synthea_ips_bh.toPandas())
export_to_vis_js(synthea_ips_nodes_pdf, synthea_ips_edges_pdf, "Synthea co-occurrence plot", "synthea_cooccurrence.html")
