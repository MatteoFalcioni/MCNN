Folder containing raw data for the model and the transforms folder for data pre-processing before training. In the following, a brief explanation of the raw data files: 

- File DTM (.ASC)
- File las originale (.las)
- File las con features (_F.las)

Lista features (oltre alle altre variabili di un normale las):
- 'ndvi'
- 'ndwi'
- 'ssi'
- 'l1_a'
- 'l2_a'
- 'l3_a'
- 'planarity_a'
- 'sphericity_a'
- 'linearity_a'
- 'entropy_a'
- 'theta_a'
- 'theta_variance_a'
- 'mad_a'
- 'delta_z_a'
- 'l1_b'
- 'l2_b'
- 'l3_b'
- 'planarity_b'
- 'sphericity_b'
- 'linearity_b'
- 'entropy_b'
- 'theta_b'
- 'theta_variance_b'
- 'mad_b'
- 'delta_z_b'
- 'N_h'
- 'delta_z_fl'

Variables ending in "_a" are computed in a neighborhood of radius 0.5m, those that end in "_b" with a radius of 1m. Those without sufixes are point-wise variables.