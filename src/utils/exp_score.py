import tensorflow as tf
from src.logging.logger import get_logger
import pandas as pd
import os
make_logger = get_logger(__name__)

def calculate_exp_score(model_name,feature_maps_dict, constant,show_exp_score = True):
    make_logger.info(f"Calculating expressivity score with constant: {constant} has been started")
    # layers_name_list = feature_maps_dict.keys()
    exp_score_dict = {}
    model_exp_score_sum = 0
    

    # num_layers = len(layers_name_list)
    # make_logger.info(f"Processing {num_layers} layers for expressivity score calculation")
    num_layers_with_scores = 0
    for layer_name, fmap in feature_maps_dict.items():
        try:
            make_logger.info('*'*100)
            make_logger.info(f"Processing layer {layer_name}")
            
            # Get feature map: shape (b, w, h, c)
            # Convert to TensorFlow tensor if it's not already
            if not isinstance(fmap, tf.Tensor):
                fmap = tf.convert_to_tensor(fmap, dtype=tf.float32)

            # Step1: Reshaping to (b*w*h, c) for 2D or 4D feature maps
            if fmap.ndim == 4:
                b, w, h, c = fmap.shape
                spatial_size = w * h
                # Reshape to (b*w*h, c)
                n= b * w * h
                fmap_2d = tf.reshape(fmap, (n, c))
                make_logger.info(f'fmap_2d shape for layer {layer_name}: {fmap_2d.shape}')
            elif fmap.ndim == 2:
                # If already 2D, assume shape is (n, c)
                n, c = fmap.shape
                spatial_size = 1
                fmap_2d = fmap
                make_logger.info(f'fmap_2d shape for layer {layer_name}: {fmap_2d.shape}')
            else:
                continue  # Skip if not 2D or 4D

            # Step 2: Center the features
            # Mean over rows (dim=0), keep dims for broadcasting
            mean = tf.reduce_mean(fmap_2d, axis=0, keepdims=True)
            make_logger.info(f'mean shape for layer {layer_name}: {mean.shape}')
            fmap_centered = fmap_2d - mean  # shape (n, c)
            make_logger.info(f'fmap_centered shape for layer {layer_name}: {fmap_centered.shape}')

            # Step 3: Calculate covariance matrix
            # Covariance matrix: (c, c)
            Covariance_matrix = tf.matmul(fmap_centered, fmap_centered, transpose_a=True) / tf.cast(tf.shape(fmap_centered)[0], tf.float32)
            make_logger.info(f'Covariance_matrix shape for layer {layer_name}: {Covariance_matrix.shape}')

            # Step 4: Compute eigenvalues (TensorFlow 2.4+)
            eigenvalues = tf.linalg.eigvalsh(Covariance_matrix) # shape (c,)
            eigenvalues = tf.where(eigenvalues <= 0, tf.maximum(eigenvalues, constant), eigenvalues)  # Ensure non-negative eigenvalues
            make_logger.info(f'eigenvalues shape for layer {layer_name}: {eigenvalues.shape}')

            # Step 5: Normalize eigenvalues to get probabilities
            prob_s = eigenvalues / tf.reduce_sum(eigenvalues)
            make_logger.info(f'prob_s shape for layer {layer_name}: {prob_s.shape}')
            # Ensure probabilities are non-negative
            prob_s = tf.where(prob_s <= 0, tf.maximum(prob_s, constant), prob_s)

            # Step 6: Calculate expressivity_score using Shanon Entropy = -Seri[prob_s * log(prob_s + constant)]
            score = -prob_s * tf.math.log(prob_s)
            expressivity_score = tf.reduce_sum(score).numpy().item()

            make_logger.info(f'Expressivity score for layer {layer_name}: {expressivity_score}')


            # Step 7: Normalize the expressivity score throughout the channels
            make_logger.info(f'Log (c): {tf.math.log(tf.cast(c, tf.float32))}')
            Normalized_expressivity_score = expressivity_score / tf.math.log(tf.cast(c, tf.float32)).numpy().item()
            make_logger.info(f'Normalized expressivity score for layer {layer_name}: {Normalized_expressivity_score}')   




            exp_score_dict[layer_name] = {"spatial_size": spatial_size, "num_channels": c,
                                      "expressivity_score": Normalized_expressivity_score}
            model_exp_score_sum += Normalized_expressivity_score

            make_logger.info(f'The score of Layer {layer_name} was added to the list')
            num_layers_with_scores += 1

        except Exception as e:
            make_logger.warning(f'N.B: The score of Layer {layer_name} was not added to the list. Exception: {e}')
            make_logger.info("*"*100)
    
    
    # Normalize the total expressivity score by the number of layers
    average_exp_score = model_exp_score_sum / num_layers_with_scores if num_layers_with_scores > 0 else 0
    make_logger.info(f"Average expressivity score for the model {model_name} is: {average_exp_score}")
    print('expressivity_score_dict:', exp_score_dict)
    print(exp_score_dict.keys())
    exp_score_df = pd.DataFrame({
        'Layer Name': list(exp_score_dict.keys()),
        'Spatial Size': [v['spatial_size'] for v in exp_score_dict.values()],
        'Number of Channels': [v['num_channels'] for v in exp_score_dict.values()],
        'Expressivity Score': [v['expressivity_score'] for v in exp_score_dict.values()],
    })
    # Add a separate row for the average expressivity score
    avg_row = pd.DataFrame({
        'Layer Name': ['Average'],
        'Spatial Size': [None],
        'Number of Channels': [None],
        'Expressivity Score': [average_exp_score],
    })
    exp_score_df = pd.concat([exp_score_df, avg_row], ignore_index=True)

    if show_exp_score:
        print(exp_score_df)
    return exp_score_df, exp_score_dict, average_exp_score