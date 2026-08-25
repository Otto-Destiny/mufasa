# Water, environment, and public-health graph test set

This is a deliberately connected sample of ten papers marked `include` by the completed Kimi benchmark run. It is designed for testing whether the graph can discover relationships across water sources, contamination, rainfall, monitoring, ecological effects, and human risk.

The PDFs are in [`pdfs`](./pdfs/).

| No. | Benchmark ID | Paper | Main graph connection | Download source |
|---:|---|---|---|---|
| 1 | GOLD-142 | Urban groundwater quality in sub-Saharan Africa: current status and implications for water security and public health | Groundwater, contamination, water security, public health | [NERC archive](https://nora.nerc.ac.uk/id/eprint/516133/1/Lapworth%20urban%20gw%20quality.pdf) |
| 2 | GOLD-149 | Comparative Analysis of Water Samples from Three Different Sources (A Case Study of Bosso, Minna Niger State) | Rainwater, wells, boreholes, drinking-water quality | [IOSR](https://www.iosrjournals.org/iosr-jestft/papers/vol4-issue5/B0450610.pdf) |
| 3 | GOLD-063 | Petroleum Hydrocarbons Contamination of Surface Water and Groundwater in the Niger Delta Region of Nigeria | Oil pollution, surface water, groundwater, health risks | [SciEP](http://pubs.sciepub.com/jephh/6/2/2/jephh-6-2-2.pdf) |
| 4 | GOLD-059 | Investigation of leachate migration using electrical resistivity imaging: a case study from an active dumpsite, Ilokun, Ado-Ekiti, Southwest Nigeria | Waste disposal, leachate, subsurface pollution, groundwater | [ITEGAM-JETIA](https://itegam-jetia.org/journal/index.php/jetia/article/download/935/633) |
| 5 | GOLD-088 | Fenthion induced toxicity and histopathological changes in gill tissue of freshwater African catfish, Clarias gariepinus | Pesticide pollution, freshwater ecosystems, aquatic toxicity | [Academic Journals](https://academicjournals.org/journal/AJB/article-full-text-pdf/A2EDAB153885.pdf) |
| 6 | GOLD-147 | Planning for a sustainable water supply through improved rainwater harvesting system in Hong Local Government Area of Adamawa State, Nigeria | Rainfall, harvesting, water supply, sustainability | [Journal DOI](http://dx.doi.org/10.31248/gjees2019.043) |
| 7 | GOLD-085 | Seasonal ARIMA Modeling and Forecasting of Rainfall in Warri Town, Nigeria | Rainfall data, forecasting, water planning | [Scientific Research Publishing](http://www.scirp.org/journal/PaperDownload.aspx?paperID=59043) |
| 8 | GOLD-055 | Analysis of Rain Rate and Rain Attenuation for Earth-Space Communication Links over Uyo, Akwa Ibom State | Rainfall measurement and modelling; cross-field use of rain data | [Nigerian Journal of Technology](https://www.nijotech.com/index.php/nijotech/article/download/1073/920/2072) |
| 9 | GOLD-086 | Comparing Single and Multiple Imputation Approaches for Missing Values in Univariate and Multivariate Water Level Data | Niger and Benue river levels, missing data, monitoring | [Strathprints](https://strathprints.strath.ac.uk/85186/1/Umar_Gray_Water_2023_Comparing_single_and_multiple_imputation_approaches_for_missing_values.pdf) |
| 10 | GOLD-089 | An assessment of flood vulnerability on physical development along drainage channels in Minna, Niger State, Nigeria | Flooding, drainage, rainfall, physical development | [Academic Journals](https://academicjournals.org/journal/AJEST/article-full-text-pdf/17F568A49514.pdf) |

Useful graph paths include `rainfall -> rainwater harvesting -> water supply`, `waste or petroleum -> water contamination -> ecological or human risk`, and `rainfall or river levels -> drainage -> flood vulnerability`.
