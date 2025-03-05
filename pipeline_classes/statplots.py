from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import linspace
from scipy.stats import norm
import matplotlib.pyplot as plt
class StatisticalPlots():
    
    @staticmethod    
    def Autocorrelation(vector, n_steps = 48, string = '', fig_size=(14,6), x=1, y=2):
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=fig_size)

        # Plot ACF
        plot_acf(vector, lags=n_steps, ax=axes[0])
        axes[0].set_title(f'{string} (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')
    
        # Plot PACF
        plot_pacf(vector, lags=n_steps, ax=axes[1])
        axes[1].set_title(f'{string} (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')
    
        plt.tight_layout()  # Adjust layout to prevent overlap
        
    

    @staticmethod
    def resid_histogram(vector, string = ''):
        '''takes resid vector as data'''
        
        plt.figure(figsize = (10,6))
        plt.hist(vector, bins= 'auto', density= True, rwidth=0.85, label=f'{string}Residuals')
        mean_resid, std_resid = norm.fit(vector)
        xmin, xmax = plt.xlim()
        curve_length = linspace(xmin, xmax, 100)
        bell_curve = norm.pdf(curve_length, mean_resid, std_resid)
        plt.title(f'{string} - Residuals')
        plt.plot(curve_length, bell_curve, 'm', linewidth = 2)
        plt.grid(axis = 'y', alpha = 0.2)
        plt.xlabel('Residuals')
        plt.ylabel('Density')

    @staticmethod
    def geoplot(df, metric = 'Occupancy', kpi = 'mean'):
        df[metric] = df.groupby(['Longitude', 'Latitude'])[metric].transform(kpi)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df[metric] , cmap='inferno', alpha=0.9)
        
        # Add color bar
        plt.colorbar(scatter, label=f'{kpi} {metric}')
        
        # Add labels and grid
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Scatter Plot of Longitude vs Latitude with {kpi} {metric}')
        plt.grid(True)

    @staticmethod
    def distribution(data, n_bins = 30, title = ''):
        # Plot a histogram using matplotlib
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=n_bins,  edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
