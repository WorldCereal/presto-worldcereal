##  <a name='WorldCerealtrainingdata'></a>WorldCereal training data

This document describes the specifications of WorldCereal training data prepared for Presto training.


#### 1. <a name='Downloadlinks'></a>Download links

Currently two versions of the training data are available:

1) With linear interpolation

- [Cropland TRAIN data (251MB)](https://artifactory.vgt.vito.be/auxdata-public/worldcereal/presto/trainingdata/annual/worldcereal_presto_cropland_linearinterp_V1_TRAIN.parquet)
- [Cropland VAL data (78MB)](https://artifactory.vgt.vito.be/auxdata-public/worldcereal/presto/trainingdata/annual/worldcereal_presto_cropland_linearinterp_V2_VAL.parquet)

2) Without linear interpolation

- [Cropland TRAIN data (242MB)](https://artifactory.vgt.vito.be/auxdata-public/worldcereal/presto/trainingdata/annual/worldcereal_presto_cropland_nointerp_V1_TRAIN.parquet)
- [Cropland VAL data (76MB)](https://artifactory.vgt.vito.be/auxdata-public/worldcereal/presto/trainingdata/annual/worldcereal_presto_cropland_nointerp_V2_VAL.parquet)


#### 2.  <a name='ContentofaWorldCerealsample'></a>Content of a WorldCereal sample

Training data comes in the form of a parquet file which can be loaded as a Pandas DataFrame, e.g.:

`df = pd.read_parquet('worldcereal_presto_cropland_linearinterp_V2_VAL.parquet')`

Let's explore the contents of one sample or row in the dataframe:

`row = df.iloc[0, :]`

We find following relevant attributes:

- Latitude/Longitude: `lat, lon`
- Start month in Presto format (note the `-1`): `datetime.strptime(start_date, "%Y-%m-%d").month - 1`
- WorldCover label: `WORLDCOVER-LABEL-10m`
- Potapov label: `POTAPOV-LABEL-10m`
- WorldCereal AEZ zone ID: `aez_zoneid`
- Land cover label: `LANDCOVER_LABEL`
- Crop type label: `CROPTYPE_LABEL`
- Irrigation label: `IRRIGATION_LABEL`
- Altitude in meters: `DEM-alt-20m`
- Slope: `DEM-slo-20m`
- WorldCereal V1 prediction label (0|1): `catboost_prediction`
- WorldCereal V1 prediction confidence (0-1): `catboost_confidence`

Monthly composited time series inputs are organized following the pattern:

`{sensor}-{band}-ts{time_step}-{resolution}m`

where:
- {sensor} is one of `OPTICAL|SAR|METEO`
- {band} are the available bands for each sensor
- {time_step} is the monthly timestep **since the start of each time series** (0-11)
- {resolution} is the original resolution of each band (though all bands and sensors have been resampled to 10m)

#### 3.  <a name='Conversioninformation'></a>Conversion information

Some things need to be taken into account for converting the raw WorldCereal data into Presto compatible inputs as outlined below. A [`WorldCerealDataset`](/src/dataset.py)class is provided to take care of these conversions using the `convert_inputs` method.

The general no data value is set at `65535`. Currently, [Presto does not allow to mask elements both in input and target](https://github.com/nasaharvest/presto/issues/26#issuecomment-1777120102). So this no data value needs to be dealt with somehow. In the **interpolated** version of the WorldCereal training data, most of these were removed by linear interpolation but occasional no data might still occur. In the **non-interpolated**, many more no data elements will occur (e.g. due to the absence of any non-clouded observation in a month).

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
