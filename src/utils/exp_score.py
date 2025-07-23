import tensorflow as tf
from src.logging.logger import get_logger
import pandas as pd
import os
make_logger = get_logger(__name__)

def calculate_exp_score(model_name,state,feature_maps_dict, constant,show_exp_score = True):
    make_logger.info(f"Calculating expressivity score with constant: {constant} has been started")
    layers_name_list = feature_maps_dict.keys()
    exp_score_dict = {}
    pca_var_dict = {}
    model_exp_score_sum = 0


    for layer in layers_name_list:
        try:
            # Get feature map: shape (b, w, h, c)
            fmap = feature_maps_dict[layer]
            
            # Convert to TensorFlow tensor if it's not already
            if not isinstance(fmap, tf.Tensor):
                fmap = tf.convert_to_tensor(fmap, dtype=tf.float32)
            
            b, w, h, c = fmap.shape
            print(f"fmap.shape", fmap.shape)

            # Reshape to (b*w*h, c)
            fmap_2d = tf.reshape(fmap, (b * w * h, c))
            print('fmap_2d',fmap_2d.shape)
            # Mean over rows (dim=0), keep dims for broadcasting
            mean = tf.reduce_mean(fmap_2d, axis=0, keepdims=True)
            print('mean',mean.shape)
            
            # Center the features
            feat_centered = fmap_2d - mean  # shape (b*w*h, c)

            # Covariance matrix: (c, c)
            Covariance_matrix = tf.matmul(feat_centered, feat_centered, transpose_a=True) / tf.cast(tf.shape(feat_centered)[0], tf.float32)
            print('Covariance_matrix',Covariance_matrix.shape)
            # Compute eigenvalues (TensorFlow 2.4+)
            eigenvalues = tf.linalg.eigvalsh(Covariance_matrix)
            print('eigenvalues',eigenvalues.shape)
            

        
            eigenvalues1 = tf.linalg.eigvalsh(Covariance_matrix)
            explained_variance = eigenvalues1 / tf.reduce_sum(eigenvalues1)
            explained_variance = tf.reverse(explained_variance, axis=[0])  # descending order
            pca_var_dict[layer]=explained_variance
            print('Here')
            # Normalize eigenvalues to get probabilities
            prob_s = eigenvalues / tf.reduce_sum(eigenvalues)

            # Calculate score = -prob_s * log(prob_s + constant)
            print("prob_s",prob_s)
            print(constant+1e+5)
            score = -prob_s * tf.math.log(prob_s + constant)
            # Sum scores to get expressivity score
            expressivity_score = tf.reduce_sum(score).numpy().item()
            
            exp_score_dict[layer] = expressivity_score
            model_exp_score_sum += expressivity_score

            make_logger.info(f'The score of Layer {layer} was added to the list')
            make_logger.info("*"*100)

        except Exception as e:
            make_logger.warning(f'N.B: The score of Layer {layer} was not added to the list. Exception: {e}')
            make_logger.info("*"*100)
    
    
    show_exp_score_df = pd.DataFrame.from_dict(exp_score_dict, orient='index', columns=['Expressivity Score'])
    show_exp_score_df.index.name = 'Layer'


    os.makedirs('exp_score_folder', exist_ok=True)
    show_exp_score_df.to_csv(f'exp_score_folder/{model_name}_{state}_expressivity_scores.csv')
    if show_exp_score:
        print(show_exp_score_df)
    return exp_score_dict, pca_var_dict, model_exp_score_sum