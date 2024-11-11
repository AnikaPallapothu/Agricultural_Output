import pandas as pd

def prepare_data(df):
    """ Separate dependent and
        independant variables 
        from data"
    
    Args
    -----
    df - training dataframe
    Returns
    -------
    ytrain - dependent variable
    xtrain - independant variable
    """
    ytrain=df['Yield_kg_per_hectare']
    xtrain=df.drop('Yield_kg_per_hectare',axis=1)

    return xtrain, ytrain

