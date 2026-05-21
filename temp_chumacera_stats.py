import pandas as pd
import numpy as np
from scipy.signal import medfilt

df = pd.read_csv('Desfibradora_crudo.csv')
cols = ['Aceleration_CHUMACERA A','Velocity_CHUMACERA A','Envelope_CHUMACERA A']
for col in cols:
    data = df[col].fillna(0).astype(float).values
    data_no_zero = data[data != 0]
    if len(data_no_zero) > 0:
        Q1 = np.percentile(data_no_zero, 25)
        Q3 = np.percentile(data_no_zero, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((data != 0) & ((data < lower) | (data > upper))).sum()
    else:
        lower = upper = 0
        outliers = 0
    data_filtered = data.copy()
    if len(data_no_zero) > 0:
        med = np.median(data_no_zero)
        mask = ((data_filtered != 0) & ((data_filtered < lower) | (data_filtered > upper)))
        data_filtered[mask] = med
    data_filtered = medfilt(data_filtered, kernel_size=5)
    print(col)
    print('CRUDO:')
    print('  Media:      {:.3f}'.format(np.mean(data)))
    print('  Std:        {:.3f}'.format(np.std(data)))
    print('  Min-Max:    {:.3f} - {:.3f}'.format(np.min(data), np.max(data)))
    print('  Outliers:   {}'.format(int(outliers)))
    print('')
    print('  FILTRADO:')
    print('  Media:      {:.3f}'.format(np.mean(data_filtered)))
    print('  Std:        {:.3f}'.format(np.std(data_filtered)))
    print('  Min-Max:    {:.3f} - {:.3f}'.format(np.min(data_filtered), np.max(data_filtered)))
    if np.mean(data) != 0:
        change = (np.mean(data_filtered) - np.mean(data)) / np.mean(data) * 100
    else:
        change = 0
    print('  Cambio:     {:+.2f}%'.format(change))
    print('')
