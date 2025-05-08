from dataclasses import dataclass
@dataclass
class EnvPaths:
    s1_Extract_RawData: str = r'..\PATH'
    s2_Transform_DataCleaned: str = r'..\PATH'
    s3_Transform_FeatureEngineered: str = r'..\PATH'
    s4_Transform_DataTransformed: str = r'..\PATH'
