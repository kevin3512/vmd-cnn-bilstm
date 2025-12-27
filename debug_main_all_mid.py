from main_all_mid import CarbonEmissionPredictor, generate_sample_data
import numpy as np

# generate small sample
data = generate_sample_data(200)

predictor = CarbonEmissionPredictor()
# reduce epochs to see quick progress
final_pred, final_actual, imfs = predictor.fit_predict(data, seq_length=24, epochs=5, batch_size=16, test_size=0.2)
print('Done quick run: pred_len=', len(final_pred), 'actual_len=', len(final_actual))