"""we love dividing by 0"""
import warnings

warnings.filterwarnings(message=r".*divide by zero", action="ignore")
warnings.filterwarnings(message=r".*invalid value enc", action="ignore")
warnings.filterwarnings(message=r".*All-NaN", action="ignore")
