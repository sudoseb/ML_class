import pandas as pd

class dfeatures:
    def __init__(self, df, origin='SE', datefield='%Date'):
        self.df = df
        self.origin = origin
        self.datefield = datefield

    def getRegular(self):
        # Use self.df instead of df
        self.df['WeekDay'] = pd.to_datetime(self.df[self.datefield]).dt.dayofweek.astype(str)
        self.df['Week'] = pd.to_datetime(self.df[self.datefield]).dt.isocalendar().week.astype(str)
        self.df['Month'] = pd.to_datetime(self.df[self.datefield]).dt.month.astype(str)
        self.df['isWeekend'] = pd.to_datetime(self.df[self.datefield]).dt.dayofweek.astype(str).apply(lambda x: '1' if x in ['5', '6'] else '0')
        return self.df

    def getHolidays(self):
        import holidays
        # Initialize holidays for the given country
        country_holidays = holidays.country_holidays(self.origin)
        
        # Use self.df instead of df
        self.df['isHoliday'] = self.df[self.datefield].apply(lambda x: '1' if x in country_holidays else '0')
        self.df['holidayName'] = self.df[self.datefield].apply(lambda x: country_holidays.get(x, 'No Holiday'))
        return self.df

    def getDateFeatures(self):
        '''
        Getting all date features - regular day, week, month as well as holidays per given origin.
        '''
        # Ensure we use the methods properly with self
        self.df = self.getHolidays()
        self.df = self.getRegular()
        return self.df

