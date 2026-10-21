# Day 1 Lab — Answers

## 1. Leakage

I dropped `mag` because it is used to create the target `big_quake` (magnitude ≥ 4.5). I also dropped `title`, `sig`, `mmi`, `cdi`, `felt`, and `alert` because they can reveal information about the earthquake's impact or severity. `tsunami` is not automatically leakage, but it may be unavailable at prediction time, so its use depends on the prediction scenario. `magType` describes the magnitude measurement method, but it should also be checked for availability and possible bias.

## 2. Stream vs. batch

During my 60-second stream window, I collected 2 rows. Both event IDs were also found in the weekly batch feed. The combined dataset contained 1 event with multiple distinct `updated` timestamps. “Latest version wins” means keeping the record with the newest `updated` timestamp when multiple versions of the same event are available.

## 3. Outliers

I would not automatically remove negative depth values because they may result from earthquake-location solutions or data-quality issues and should be investigated first. I would also not automatically remove negative magnitudes because small earthquakes can have negative magnitudes. I would check the USGS data documentation and use validation rules or flags rather than removing valid observations without evidence.

## 4. Cardinality

I grouped the `region` feature so that the 15 most frequent values remain separate and less frequent values become `Other`. One alternative is frequency encoding, which replaces each category with its frequency in the training data. Its risk is that different regions may receive similar values, and the encoding must be learned from training data only to avoid data leakage.

