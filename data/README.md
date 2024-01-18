##  <a name='WorldCerealdata'></a>WorldCereal benchmarking data

This document describes the specifications of WorldCereal benchmarking data .

The benchmarking data is available on [OneDrive](https://vitoresearch-my.sharepoint.com/:f:/g/personal/kristof_vantricht_vito_be/Eq8ElrvwZzFNoUTVByzEiowBUc2fWhgWfAxMCdsNhf3V1g) (permission required).


#### 1. <a name='FileDescription'></a>File Description

The following files are currently available:

- **rawts-monthly_calval.parquet (309MB)**
  This dataframe contains the original monthly (masked) time series that can be used as an input to Presto or to compute classification features manually.
- **rawts-10d_calval.parquet (824MB)**
  This dataframe contains the original dekadal (10-day) (masked) time series that can be used as an input to *modified* Presto or to compute classification features manually.
- **expertfts-10d_calval.parquet (501MB)**
  This dataframe contains the official WorldCereal expert features that were used in V1 of the crop/no-crop WorldCereal product, computed for `rawts-10d_calval.parquet`.
- **prestofts-monthly_calval.parquet (566MB)**
  This dataframe contains the Presto embeddings using the currently best-performing finetuned Presto model on WorldCereal data. These embeddings were computed for `rawts-monthly_calval.parquet`.

As the name suggests, the files are a (shuffled) concatenation of the original **CAL** and **VAL** files. Benchmarking experiments should now do their own split based on certain scenarios. It's crucial to keep track of how this split is done for different scenarios.

The parquet files mentioned above are now fully aligned and contain the exact same samples (pixels), identified by as unique `sample_id`. Any dynamic generation of a hold-out set should be based on this `sample_id` so other experiments using the other files can use the same splitting strategy.


#### 2.  <a name='ContentofaWorldCerealsample'></a>Content of a WorldCereal sample

Training data comes in the form of a parquet file which can be loaded as a Pandas DataFrame, e.g.:

`df = pd.read_parquet('rawts-monthly_calval.parquet')`

Let's explore the contents of one sample or row in the dataframe:

`row = df.iloc[0, :]`

We find following relevant attributes:

- Unique identifier of a sample: `sample_id`
- Identifier of the dataset to which the sample belongs: `ref_id`
- Latitude/Longitude: `lat, lon`
- Year to which the sample belongs: `year`
- Validity date (e.g. observation date) of the sample: `valid_date`
- Start date of the time series: `start_date`
- End date of the time series: `end_date`
- WorldCover label: `WORLDCOVER-LABEL-10m`
- Potapov label: `POTAPOV-LABEL-10m`
- WorldCereal AEZ zone ID: `aez_zoneid`
- Land cover label: `LANDCOVER_LABEL`
- Crop type label: `CROPTYPE_LABEL`
- Altitude in meters: `DEM-alt-20m`
- Slope: `DEM-slo-20m`
- WorldCereal V1 prediction label (0|1): `worldcereal_prediction`
- WorldCereal V1 prediction confidence (0-1): `worldcereal_confidence`

Presto requires the start month. This can be computed using (note the `-1`): `datetime.strptime(start_date, "%Y-%m-%d").month - 1`

Monthly composited time series inputs are organized following the pattern:

`{sensor}-{band}-ts{time_step}-{resolution}m`

where:
- {sensor} is one of `OPTICAL|SAR|METEO`
- {band} are the available bands for each sensor
- {time_step} is the monthly timestep **since the start of each time series** (0-11)
- {resolution} is the original resolution of each band (though all bands and sensors have been resampled to 10m)

Note that `expertfts` and `prestofts` dataframes have already computed features/embeddings on these time series. Their names should be clear.
Note also that for 10D data, the time series span 36 steps for the entire year.

#### 3.  <a name='Conversioninformation'></a>Conversion information

Some things need to be taken into account for converting the raw WorldCereal data into Presto compatible inputs as outlined below. A [`WorldCerealDataset`](/src/dataset.py)class is provided to take care of these conversions using the `convert_inputs` method.

The general no data value is set at `65535`. [Presto now supports a real mask](https://github.com/WorldCereal/presto-worldcereal/pull/11) and the data loader is configured to pass this no data through as masked values.

##### 3.1  <a name='OpticalData'></a>Optical data
Optical data is in scaled radiances (scalefactor 10,000) and is therefore directly compatible with Presto.

##### 3.2  <a name='SARData'></a>SAR data
SAR backscatter data is in scaled dB and needs to be unscaled to true dB to be compatible with Presto:

```
# Conversion to dB
true_dB = 20 * np.log10(scaled_dB) - 83
```

##### 3.3  <a name='MeteoData'></a>Meteo data
METEO variables are scaled (scale factor 100) and need to be unscaled to be compatible with Presto.
Specifically for the precipitation sum, values are (after unscaling) expressed in `mm` and need to be converted to `m` to be compatible with Presto.
