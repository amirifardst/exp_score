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
    entropy_dict = {}
    model_exp_score_sum = 0
    

    num_layers = len(layers_name_list)
    make_logger.info(f"Processing {num_layers} layers for expressivity score calculation")
    num_layers_with_scores = 0
    for i, layer in enumerate(layers_name_list):
        try:
            make_logger.info('*'*100)
            make_logger.info(f"Processing layer {i+1}: {layer}")
            
            # Get feature map: shape (b, w, h, c)
            fmap = feature_maps_dict[layer]
            make_logger.info(f'fmap shape for layer {layer}: {fmap.shape}')

            
            # Convert to TensorFlow tensor if it's not already
            if not isinstance(fmap, tf.Tensor):
                fmap = tf.convert_to_tensor(fmap, dtype=tf.float32)
            
         
            b, w, h, c = fmap.shape

            # Reshape to (b*w*h, c)
            n= b * w * h
            fmap_2d = tf.reshape(fmap, (n, c))
            make_logger.info(f'fmap_2d shape for layer {layer}: {fmap_2d.shape}')

            # Mean over rows (dim=0), keep dims for broadcasting
            mean = tf.reduce_mean(fmap_2d, axis=0, keepdims=True)
            make_logger.info(f'mean shape for layer {layer}: {mean.shape}')

            # Center the features
            fmap_centered = fmap_2d - mean  # shape (b*w*h, c)
            make_logger.info(f'fmap_centered shape for layer {layer}: {fmap_centered.shape}')

            # Covariance matrix: (c, c)
            Covariance_matrix = tf.matmul(fmap_centered, fmap_centered, transpose_a=True) / tf.cast(tf.shape(fmap_centered)[0], tf.float32)
            make_logger.info(f'Covariance_matrix shape for layer {layer}: {Covariance_matrix.shape}')

            # Compute eigenvalues (TensorFlow 2.4+)
            eigenvalues = tf.linalg.eigvalsh(Covariance_matrix)
            make_logger.info(f'eigenvalues shape for layer {layer}: {eigenvalues.shape}')

            # Normalize eigenvalues to get probabilities
            prob_s = eigenvalues / tf.reduce_sum(eigenvalues)
            make_logger.info(f'prob_s shape for layer {layer}: {prob_s.shape}')
            # Ensure probabilities are non-negative
            prob_s = tf.where(prob_s <= 0, tf.maximum(prob_s, constant), prob_s)

            # Calculate expressivity_score(Entropy) = -Seri[prob_s * log(prob_s + constant)]
            score = -prob_s * tf.math.log(prob_s)
            expressivity_score = tf.reduce_sum(score).numpy().item()

            make_logger.info(f'Expressivity score for layer {layer}: {expressivity_score}')


            # Normalize the expressivity score throughout the channels
            make_logger.info(f'Log (c): {tf.math.log(tf.cast(c, tf.float32))}')
            Normalized_expressivity_score = expressivity_score / tf.math.log(tf.cast(c, tf.float32)).numpy().item()
            entropy = Normalized_expressivity_score
            entropy_dict[layer] = entropy
            make_logger.info(f'Normalized expressivity score for layer {layer}: {Normalized_expressivity_score}')

            

            
            eigenvalues1 = tf.linalg.eigvalsh(Covariance_matrix)
            explained_variance = eigenvalues1 / tf.reduce_sum(eigenvalues1)
            explained_variance = tf.reverse(explained_variance, axis=[0])  # descending order
            pca_var_dict[layer]=explained_variance
           


           
            
            
            exp_score_dict[layer] = Normalized_expressivity_score
            model_exp_score_sum += Normalized_expressivity_score

            make_logger.info(f'The score of Layer {layer} was added to the list')
            num_layers_with_scores += 1

        except Exception as e:
            make_logger.warning(f'N.B: The score of Layer {layer} was not added to the list. Exception: {e}')
            make_logger.info("*"*100)
    
    
    # Normalize the total expressivity score by the number of layers
    average_exp_score = model_exp_score_sum / num_layers_with_scores if num_layers_with_scores > 0 else 0
    make_logger.info(f"Average expressivity score for the model {model_name} is: {average_exp_score}")
    print('expressivity_score_dict:', exp_score_dict)
    exp_score_df = pd.DataFrame({
        'Layer Name': list(exp_score_dict.keys()),
        'Expressivity Score': list(exp_score_dict.values()),
    })
    # Add a separate row for the average expressivity score
    avg_row = pd.DataFrame({
        'Layer Name': ['Average'],
        'Expressivity Score': [average_exp_score],
    })
    exp_score_df = pd.concat([exp_score_df, avg_row], ignore_index=True)

    if show_exp_score:
        print(exp_score_df)
    return exp_score_df, exp_score_dict,entropy_dict, pca_var_dict, average_exp_score