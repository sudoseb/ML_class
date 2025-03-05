from dataclasses import dataclass
#### PLACE HOLDER
@dataclass
class EnvPaths:
    s1_Extract_RawData: str = r'..\TF_ETL\transactions.parquet'
    s2_Transform_DataCleaned: str = r'..\TF_ETL\Cleaned_df.parquet'
    s3_Transform_FeatureEngineered: str = r'..\TF_ETL\baseline_test_frames//'
    s4_Load_DataTransformed: str = r'..\TF_ETL\finalfiles\\final.parquet'
